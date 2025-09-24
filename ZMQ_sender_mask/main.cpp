// main.cpp  ZMQ [json_meta, mask_bytes] + OpenGL draw + GPIO sync + L-frame FIFO + CSV mapper
// Overlay shows two rows of big digits: top = running counter starting at 1, bottom = mask id or proj index.
// CSV saved in CWD as "mask_map.csv": each line = "mask_id,cam_idx"
//
// Build: g++ -O2 -std=c++17 main.cpp -o projector -lglfw -lGL -lzmq -lgpiod -lpthread

#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <gpiod.h>
#include <zmq.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cmath>

#define WIDTH  1920
#define HEIGHT 1080

// ---------- util ----------
static inline int64_t now_ns(){
    timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (int64_t)ts.tv_sec*1000000000LL + (int64_t)ts.tv_nsec;
}

// simple thread safe log
static std::mutex g_log_mtx;
template <typename... Args>
static void LOG(Args&&... args){
    std::ostringstream oss; (oss << ... << std::forward<Args>(args));
    std::lock_guard<std::mutex> lk(g_log_mtx);
    std::cout << oss.str() << std::flush;
}

// ---------- config, CLI overridable ----------
enum class Edge { Rising, Falling, Both };
static const char* edge_name(Edge e){
    switch(e){ case Edge::Rising: return "rising"; case Edge::Falling: return "falling"; default: return "both"; }
}

static std::string PROJ_TRIG_CHIP = "/dev/gpiochip1";
static int         PROJ_TRIG_LINE = 9;
static Edge        PROJ_EDGE      = Edge::Rising;

static std::string CAM_TRIG_CHIP  = "/dev/gpiochip1";
static int         CAM_TRIG_LINE  = 8;
static Edge        CAM_EDGE       = Edge::Rising;

static int         LATENCY_FRAMES = 1;
static std::string ZMQ_BIND       = "tcp://*:5558";
static int         SWAP_INTERVAL  = 1;        // 0 no vsync, 1 vsync
static int         MONITOR_PICK   = 1;        // -1 pick rightmost, else exact index
static bool        VISIBLE_ID     = true;     // draw overlay

// overlay options
// OVERLAY_STYLE: 0 barcode, 1 digits
static int  OVERLAY_STYLE = 1;
static int  OVERLAY_CELL  = 12;     // scale unit in pixels (smaller digits)
static int  OVERLAY_OFF_X = 520;    // offset from left in pixels
static int  OVERLAY_OFF_Y = 380;    // offset from top in pixels
static bool OVERLAY_BG    = true;   // black background plate

// bottom row mode: 0 mask id, 1 proj index, 2 none
static int OVERLAY_BOTTOM_MODE = 0;

// mapping options
static int64_t CAM_TS_OFFSET_US = 0;   // shift applied to camera trigger timestamp before mapping
static int64_t MAP_EPS_US       = 500; // tolerance window for mapping jitter
static int      CAM_WARMUP      = 10;  // number of initial cam triggers to treat as warm-up

// CSV output (always saved as "mask_map.csv" in CWD unless overridden)
static std::string MAP_CSV_PATH = "mask_map.csv";

// ---------- homography (H) reception and mapping ----------
static std::string ZMQ_H_BIND   = "tcp://*:5560"; // REP endpoint to receive 3x3 H (float64[9])
static int         HORIZ_FLIP   = 1;              // 1 = mirror horizontally after warp (to match Python path)
static std::string H_FILE_PATH  = "";            // optional on-disk H preload (text with 9 doubles)

static std::mutex           g_h_mtx;
static double               g_H[9]       = {1,0,0, 0,1,0, 0,0,1};
static std::vector<int>     g_h_src_idx;          // size WIDTH*HEIGHT, maps dst idx -> src idx (-1 if oob)
static std::atomic<bool>    g_h_ready{false};

// ---------- shared state ----------
static std::atomic<bool> g_running{true};

// ZMQ -> camera
static std::atomic<int> latest_mask_id{-1};

// Camera FIFO for L-frame aging
static std::deque<int> cam_fifo;
static std::mutex      cam_fifo_mtx;

// ----- NEW: lock-protected FIFO for ready ids (camera->projector) -----
struct IntQueue {
    std::deque<int> q;
    std::mutex m;
    size_t capacity = 4096;  // prevent unbounded growth

    void push(int v){
        std::lock_guard<std::mutex> lk(m);
        if (q.size() >= capacity) q.pop_front(); // drop oldest if overflow
        q.push_back(v);
    }
    bool try_pop(int& out){
        std::lock_guard<std::mutex> lk(m);
        if (q.empty()) return false;
        out = q.front(); q.pop_front(); return true;
    }
    size_t size(){
        std::lock_guard<std::mutex> lk(m);
        return q.size();
    }
};

static IntQueue g_ready_q;    // what to DRAW next (camera-aged masks)
// ----- NEW: lock-protected FIFO for swapped ids (renderer->projector) -----
static IntQueue g_swapped_q;  // what actually got SWAPPED (will be visible on *this* projector frame)

// Projector-visible bookkeeping
static std::atomic<uint64_t> cam_frame_idx{0};
static std::atomic<uint64_t> proj_trig_idx{0};
static std::atomic<int>      last_visible_mask_id{-1};  // actually visible on last pidx
static std::atomic<uint64_t> last_visible_proj_idx{0};
// Camera trigger overlay/mapping helpers
static std::atomic<int>      camera_trigger_count{0};
static std::atomic<uint64_t> last_cam_idx{0};
static std::atomic<uint64_t> last_matched_proj_for_cam{0};

// running draw counter, starts at 1 and increments each drawn frame
static std::atomic<uint64_t> draw_counter{0};

// notify main thread to draw a specific id and annotate with the pidx it will target (next frame)
static std::atomic<int>      pending_draw_id{-1};
static std::atomic<uint64_t> pending_draw_proj_idx{0};
static GLFWwindow*           g_win = nullptr;

// Mask cache, keyed by id
struct MaskCache {
    std::unordered_map<int, std::vector<unsigned char>> map;
    std::deque<int> order;               // insertion order for simple eviction
    size_t capacity = 512;               // simple cap
    std::mutex mtx;

    void put(int id, const unsigned char* bytes, size_t n){
        std::lock_guard<std::mutex> lk(mtx);
        auto it = map.find(id);
        if (it == map.end()){
            if (order.size() >= capacity){
                int evict = order.front(); order.pop_front();
                map.erase(evict);
            }
            order.push_back(id);
            map.emplace(id, std::vector<unsigned char>(bytes, bytes + n));
        } else {
            it->second.assign(bytes, bytes + n);
        }
    }
    bool get(int id, const unsigned char*& ptr, size_t& n){
        std::lock_guard<std::mutex> lk(mtx);
        auto it = map.find(id);
        if (it == map.end()) return false;
        ptr = it->second.data(); n = it->second.size();
        return true;
    }
} g_cache;

// ---------- projector trigger history for mapping ----------
struct ProjEvent {
    uint64_t pidx;
    int64_t  t_ns;
    int      mask_id;   // the mask actually visible for this projector frame
};
static std::mutex             proj_hist_mtx;
static std::deque<ProjEvent>  proj_hist;   // append-only; keep a few seconds worth
static const size_t           PROJ_HIST_MAX = 4096;

// ---------- OpenGL draw ----------
static void draw_mask_pixels(const void* data, int w, int h){
    // Fast path: state changes minimized for each frame
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelZoom(1.0f, -1.0f);               // flip vertical, top left origin masks
    glRasterPos2f(-1.f, 1.f);               // top left
    glDrawPixels(w, h, GL_LUMINANCE, GL_UNSIGNED_BYTE, data);
}

// ---------- homography helpers ----------
static bool invert_3x3(const double M[9], double Inv[9]){
    double a = M[0], b = M[1], c = M[2];
    double d = M[3], e = M[4], f = M[5];
    double g = M[6], h = M[7], i = M[8];
    double A =   (e*i - f*h);
    double B = - (d*i - f*g);
    double C =   (d*h - e*g);
    double D = - (b*i - c*h);
    double E =   (a*i - c*g);
    double F = - (a*h - b*g);
    double G =   (b*f - c*e);
    double H = - (a*f - c*d);
    double I =   (a*e - b*d);
    double det = a*A + b*B + c*C;
    if (std::fabs(det) < 1e-12) return false;
    double invdet = 1.0 / det;
    Inv[0] = A * invdet; Inv[1] = D * invdet; Inv[2] = G * invdet;
    Inv[3] = B * invdet; Inv[4] = E * invdet; Inv[5] = H * invdet;
    Inv[6] = C * invdet; Inv[7] = F * invdet; Inv[8] = I * invdet;
    return true;
}

static void precompute_h_map_unlocked(){
    // pre: g_H already set, lock held
    double Hin[9];
    if (!invert_3x3(g_H, Hin)){
        g_h_src_idx.clear();
        g_h_ready.store(false);
        LOG("[HMAP] singular H, disabled mapping\n");
        return;
    }
    g_h_src_idx.assign((size_t)WIDTH * (size_t)HEIGHT, -1);
    const int W = WIDTH, Ht = HEIGHT;
    for (int y = 0; y < Ht; ++y){
        for (int x = 0; x < W; ++x){
            // Optional horizontal flip at display stage (match Python path)
            int xd = HORIZ_FLIP ? (W - 1 - x) : x;
            double X = (double)xd;
            double Y = (double)y;
            double denom = Hin[6]*X + Hin[7]*Y + Hin[8];
            if (std::fabs(denom) < 1e-12){
                continue;
            }
            double xs = (Hin[0]*X + Hin[1]*Y + Hin[2]) / denom;
            double ys = (Hin[3]*X + Hin[4]*Y + Hin[5]) / denom;
            int xi = (int)std::llround(xs);
            int yi = (int)std::llround(ys);
            if (xi >= 0 && xi < W && yi >= 0 && yi < Ht){
                size_t dst_idx = (size_t)y * (size_t)W + (size_t)x;
                size_t src_idx = (size_t)yi * (size_t)W + (size_t)xi;
                g_h_src_idx[dst_idx] = (int)src_idx;
            }
        }
    }
    g_h_ready.store(true);
    LOG("[HMAP] precomputed mapping (", W, "x", Ht, ")\n");
}

static void warp_mask_nn(const unsigned char* src, std::vector<unsigned char>& dst){
    if (!src){
        dst.assign((size_t)WIDTH * (size_t)HEIGHT, 0);
        return;
    }
    const size_t N = (size_t)WIDTH * (size_t)HEIGHT;
    if (dst.size() != N) dst.resize(N);
    // No lock here: we only read g_h_src_idx which is replaced atomically under lock before ready flag set
    for (size_t i = 0; i < N; ++i){
        int si = (i < g_h_src_idx.size()) ? g_h_src_idx[i] : -1;
        dst[i] = (si >= 0) ? src[(size_t)si] : 0;
    }
}

static bool load_h_from_text_file(const std::string& path){
    std::ifstream f(path.c_str());
    if (!f.is_open()){
        LOG("[HMAP] cannot open H file ", path, "\n");
        return false;
    }
    double vals[9];
    for (int k=0;k<9;++k){
        if (!(f >> vals[k])){
            LOG("[HMAP] failed to read 9 doubles from ", path, "\n");
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_h_mtx);
        for (int k=0;k<9;++k) g_H[k] = vals[k];
        precompute_h_map_unlocked();
    }
    LOG("[HMAP] preloaded H from ", path, "\n");
    return true;
}

// ---------- overlay builders ----------
static const uint16_t DIGIT_3x5[10] = {
    0b111101101101111, // 0
    0b010110010010111, // 1
    0b111001111100111, // 2
    0b111001111001111, // 3
    0b101101111001001, // 4
    0b111100111001111, // 5
    0b111100111101111, // 6
    0b111001001001001, // 7
    0b111101111101111, // 8
    0b111101111001111  // 9
};

static inline void blit_rect(std::vector<unsigned char>& out, int ow,
                             int x, int y, int w, int h, unsigned char v){
    if (w <= 0 || h <= 0) return;
    for (int yy = 0; yy < h; ++yy){
        std::memset(&out[(y + yy) * ow + x], v, w);
    }
}

static void draw_digit_3x5(std::vector<unsigned char>& out, int ow,
                           int px, int py, int cell, int d, unsigned char v){
    if (d < 0 || d > 9) return;
    uint16_t pat = DIGIT_3x5[d];
    for (int r = 0; r < 5; ++r){
        for (int c = 0; c < 3; ++c){
            int bit = r * 3 + c;
            if ((pat >> bit) & 1){
                blit_rect(out, ow, px + c*cell, py + r*cell, cell, cell, v);
            }
        }
    }
}

static void draw_number_row(std::vector<unsigned char>& out, int ow,
                            int start_x, int start_y, int cell,
                            const std::string& s, unsigned char v){
    const int digit_w = 3*cell, digit_h = 5*cell, gap = cell;
    int x = start_x;
    for (char ch : s){
        if (ch >= '0' && ch <= '9'){
            draw_digit_3x5(out, ow, x, start_y, cell, ch - '0', v);
            x += digit_w + gap;
        } else if (ch == ' '){
            x += digit_w + gap;
        }
    }
}

static void build_overlay_digits(uint64_t counter, const std::string& bottom, int cell,
                                 std::vector<unsigned char>& out, int& ow, int& oh)
{
    std::string top_s = std::to_string(counter);
    std::string bot_s = bottom;

    const int digit_w = 3*cell, digit_h = 5*cell, gap = cell;
    const int pad = cell;
    const int rows = bot_s.empty() ? 1 : 2;
    const int row_gap = 2*cell;

    int top_w  = (int)top_s.size() * (digit_w + gap) - gap;
    int bot_w  = bot_s.empty() ? 0 : (int)bot_s.size() * (digit_w + gap) - gap;
    int text_w = std::max(top_w, bot_w);
    ow = text_w + 2*pad;
    oh = digit_h + 2*pad + (rows == 2 ? (row_gap + digit_h) : 0);

    out.assign(ow * oh, 0);

    int x0_top = pad + (text_w - top_w)/2;
    int y0_top = pad;
    draw_number_row(out, ow, x0_top, y0_top, cell, top_s, 255);

    if (!bot_s.empty()){
        int x0_bot = pad + (text_w - bot_w)/2;
        int y0_bot = pad + digit_h + row_gap;
        draw_number_row(out, ow, x0_bot, y0_bot, cell, bot_s, 255);
    }
}

static void build_overlay_barcode(uint8_t id8, uint8_t p8, uint8_t hb8,
                                  int cell,
                                  std::vector<unsigned char>& out, int& ow, int& oh)
{
    const int cells_x = 6, cells_y = 4;
    ow = cells_x * cell; oh = cells_y * cell;
    out.assign(ow * oh, 0);

    uint32_t bits = 0;
    bits |= uint32_t(id8);
    bits |= uint32_t(p8)  << 8;
    bits |= uint32_t(hb8) << 16;

    int b = 0;
    for (int y = 0; y < cells_y; ++y){
        for (int x = 0; x < cells_x; ++x, ++b){
            unsigned char val = ((bits >> b) & 1) ? 255 : 0;
            int x0 = x * cell, y0 = y * cell;
            for (int yy = 0; yy < cell; ++yy){
                std::memset(&out[(y0 + yy) * ow + x0], val, cell);
            }
        }
    }
}

// Draw overlay with ortho projection and optional black plate
static void draw_overlay_pixels(const unsigned char* px, int ow, int oh, int offX, int offY){
    if (!g_win || !px) return;
    int winW=0, winH=0; glfwGetFramebufferSize(g_win, &winW, &winH);

    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelZoom(1.0f, 1.0f);

    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity();
    glOrtho(0, winW, 0, winH, -1, 1);
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();

    int x = offX;
    int y = winH - offY - oh;

    if (OVERLAY_BG){
        int m = 4;
        glColor3f(0.f, 0.f, 0.f);
        glBegin(GL_QUADS);
        glVertex2i(x - m,     y - m);
        glVertex2i(x + ow + m, y - m);
        glVertex2i(x + ow + m, y + oh + m);
        glVertex2i(x - m,     y + oh + m);
        glEnd();
        glColor3f(1.f, 1.f, 1.f);
    }

    glRasterPos2i(x, y);
    glDrawPixels(ow, oh, GL_LUMINANCE, GL_UNSIGNED_BYTE, px);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ---------- overlay composition helpers (CPU) ----------
static void blit_onto_fullscreen(std::vector<unsigned char>& full, int W, int H,
                                 const std::vector<unsigned char>& small, int ow, int oh,
                                 int offX, int offY){
    if (ow <= 0 || oh <= 0) return;
    for (int y = 0; y < oh; ++y){
        int dy = offY + y;
        if (dy < 0 || dy >= H) continue;
        int sy = y;
        int sx0 = 0;
        int dx0 = offX;
        int run = ow;
        // clip left
        if (dx0 < 0){
            sx0 -= dx0; run += dx0; dx0 = 0;
        }
        // clip right
        if (dx0 + run > W){
            run = W - dx0;
        }
        if (run <= 0) continue;
        const unsigned char* src = small.data() + (size_t)sy * (size_t)ow + (size_t)sx0;
        unsigned char* dst = full.data() + (size_t)dy * (size_t)W + (size_t)dx0;
        std::memcpy(dst, src, (size_t)run);
    }
}

static void composite_overlay_cpu(std::vector<unsigned char>& base, // full-screen
                                  const std::vector<unsigned char>& small, int ow, int oh,
                                  int offX, int offY,
                                  bool apply_black_plate){
    const int W = WIDTH, H = HEIGHT;
    if (ow <= 0 || oh <= 0) return;
    if ((int)base.size() != W*H) base.resize((size_t)W*(size_t)H);

    // Optional: black plate behind overlay area (unwarped path)
    if (apply_black_plate){
        for (int y = 0; y < oh; ++y){
            int dy = offY + y;
            if (dy < 0 || dy >= H) continue;
            int dx0 = offX;
            int run = ow;
            if (dx0 < 0){ run += dx0; dx0 = 0; }
            if (dx0 + run > W){ run = W - dx0; }
            if (run <= 0) continue;
            unsigned char* dst = base.data() + (size_t)dy * (size_t)W + (size_t)dx0;
            std::memset(dst, 0, (size_t)run);
        }
    }
    // Bright digits: max blend
    for (int y = 0; y < oh; ++y){
        int dy = offY + y;
        if (dy < 0 || dy >= H) continue;
        int sx0 = 0;
        int dx0 = offX;
        int run = ow;
        if (dx0 < 0){ sx0 -= dx0; run += dx0; dx0 = 0; }
        if (dx0 + run > W){ run = W - dx0; }
        if (run <= 0) continue;
        const unsigned char* src = small.data() + (size_t)y * (size_t)ow + (size_t)sx0;
        unsigned char* dst = base.data() + (size_t)dy * (size_t)W + (size_t)dx0;
        for (int i = 0; i < run; ++i){ dst[i] = std::max(dst[i], src[i]); }
    }
}

// ---------- GPIO helpers ----------
static gpiod_line* request_edge_line(const std::string& chip_path, int line, Edge e, const char* tag){
    gpiod_chip* chip = gpiod_chip_open(chip_path.c_str());
    if (!chip){ LOG("[ERR ] open chip failed ", chip_path, "\n"); return nullptr; }
    gpiod_line* l = gpiod_chip_get_line(chip, line);
    if (!l){ LOG("[ERR ] get line failed ", chip_path, ":", line, "\n"); gpiod_chip_close(chip); return nullptr; }
    int rc = -1;
    if      (e == Edge::Rising)  rc = gpiod_line_request_rising_edge_events(l, tag);
    else if (e == Edge::Falling) rc = gpiod_line_request_falling_edge_events(l, tag);
    else                         rc = gpiod_line_request_both_edges_events(l, tag);
    if (rc < 0){ LOG("[ERR ] request events failed on ", chip_path, ":", line, "\n"); gpiod_chip_close(chip); return nullptr; }
    return l;
}

// ---------- tiny JSON id parser ----------
static int parse_id_from_json(const std::string& s, int fallback){
    try { size_t pos = 0; int v = std::stoi(s, &pos); if (pos == s.size()) return v; } catch(...) {}
    size_t p = s.find("\"id\""); if (p == std::string::npos) p = s.find("'id'");
    if (p == std::string::npos) return fallback;
    p = s.find_first_of("0123456789-+", p);
    if (p == std::string::npos) return fallback;
    try { return std::stoi(s.c_str() + p); } catch(...) { return fallback; }
}

static bool parse_flag_from_json(const std::string& s, const char* key){
    size_t p = s.find(key);
    if (p == std::string::npos) return false;
    p = s.find_first_of("0123456789tT", p);
    if (p == std::string::npos) return false;
    if (s[p] == '1') return true;
    if ((p+3) < s.size()){
        char c0 = s[p], c1 = s[p+1], c2 = s[p+2], c3 = s[p+3];
        if ((c0=='t'||c0=='T') && (c1=='r'||c1=='R') && (c2=='u'||c2=='U') && (c3=='e'||c3=='E')) return true;
    }
    return false;
}

static int parse_opt_bool_from_json(const std::string& s, const char* key){
    size_t p = s.find(key);
    if (p == std::string::npos) return -1; // not present
    p = s.find_first_of("0123456789tTfF", p);
    if (p == std::string::npos) return -1;
    if (s[p] == '1') return 1;
    if (s[p] == '0') return 0;
    auto lower = [](char c){ return (char)((c>='A'&&c<='Z')? (c-'A'+'a') : c); };
    if ((p+3) < s.size()){
        char c0 = lower(s[p]), c1 = lower(s[p+1]), c2 = lower(s[p+2]), c3 = lower(s[p+3]);
        if (c0=='t' && c1=='r' && c2=='u' && c3=='e') return 1;
    }
    if ((p+4) < s.size()){
        char c0 = lower(s[p]), c1 = lower(s[p+1]), c2 = lower(s[p+2]), c3 = lower(s[p+3]), c4 = lower(s[p+4]);
        if (c0=='f' && c1=='a' && c2=='l' && c3=='s' && c4=='e') return 0;
    }
    return -1;
}

// ---------- CLI helpers ----------
static inline std::string trim(const std::string& s){
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    if (b == std::string::npos) return std::string();
    return s.substr(b, e - b + 1);
}
static int safe_stoi(const std::string& s_in, int def, const char* name){
    std::string s = trim(s_in);
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str()){
        LOG("[CLI ] bad integer for ", name, " value '", s_in, "', using ", def, "\n");
        return def;
    }
    return (int)v;
}
static long long safe_stoll(const std::string& s_in, long long def, const char* name){
    std::string s = trim(s_in);
    char* end = nullptr;
    long long v = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str()){
        LOG("[CLI ] bad integer for ", name, " value '", s_in, "', using ", def, "\n");
        return def;
    }
    return v;
}
static void parse_pos_pair(const std::string& v, int& x, int& y){
    auto t = trim(v);
    auto c = t.find(',');
    if (c == std::string::npos){
        LOG("[CLI ] bad --overlay-pos, expected X,Y got '", v, "'\n");
        return;
    }
    x = safe_stoi(t.substr(0, c), x, "overlay-pos.x");
    y = safe_stoi(t.substr(c+1),  y, "overlay-pos.y");
}

// ---------- threads ----------
static void zmq_thread_func(){
    zmq::context_t ctx(1);
    zmq::socket_t  sock(ctx, ZMQ_PULL);

    int rcvtimeo = 200; sock.setsockopt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
    int linger   = 0;   sock.setsockopt(ZMQ_LINGER,   &linger,   sizeof(linger));

    try { sock.bind(ZMQ_BIND); }
    catch (const zmq::error_t& e){ LOG("[ERR ] ZMQ bind failed ", ZMQ_BIND, " ", e.what(), "\n"); return; }
    LOG("Listening on ", ZMQ_BIND, "\n");

    const size_t expected = size_t(WIDTH) * size_t(HEIGHT);

    while (g_running.load()){
        zmq::message_t part1;
        auto ok1 = sock.recv(part1, zmq::recv_flags::none);
        if (!ok1) continue;

        int more = 0; size_t moresz = sizeof(more);
        sock.getsockopt(ZMQ_RCVMORE, &more, &moresz);
        if (!more){ LOG("[ZMQ ] expected multipart, got one part\n"); continue; }

        zmq::message_t part2;
        auto ok2 = sock.recv(part2, zmq::recv_flags::none);
        if (!ok2){ LOG("[ZMQ ] failed second part\n"); continue; }

        sock.getsockopt(ZMQ_RCVMORE, &more, &moresz);
        if (more){
            zmq::message_t dummy;
            while (sock.getsockopt(ZMQ_RCVMORE, &more, &moresz), more){
                auto rc = sock.recv(dummy, zmq::recv_flags::none);
                if (!rc) break;
            }
        }

        std::string meta(static_cast<const char*>(part1.data()), part1.size());
        int id_prev = latest_mask_id.load();
        int id = parse_id_from_json(meta, id_prev < 0 ? 1 : id_prev + 1);

        if (part2.size() != expected){
            LOG("[ZMQ ] bad mask size ", part2.size(), ", expected ", expected, "\n");
            continue;
        }

        g_cache.put(id, static_cast<const unsigned char*>(part2.data()), part2.size());
        latest_mask_id.store(id);

        // If client requests immediate scheduling (pattern mode), enqueue for next projector frame
        bool immediate = parse_flag_from_json(meta, "immediate");
        if (immediate){
            // Clear stale queued ids to reduce latency/jitter for burst pattern streams
            {
                while (g_ready_q.size() > 4){ int drop = -1; g_ready_q.try_pop(drop); }
            }
            g_ready_q.push(id);
        }

        // Optional runtime overlay toggle
        int vis = parse_opt_bool_from_json(meta, "visible_id");
        if (vis >= 0){ VISIBLE_ID = (vis != 0); }

        LOG("[ZMQ ] received id=", id, immediate?" (immediate)": "", ", cached ", part2.size(), " bytes\n");
    }

    sock.close();
    ctx.close();
}

static void h_zmq_thread_func(){
    zmq::context_t ctx(1);
    zmq::socket_t  rep(ctx, ZMQ_REP);
    int rcvtimeo = 200; rep.setsockopt(ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
    int sndtimeo = 200; rep.setsockopt(ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
    int linger   = 0;   rep.setsockopt(ZMQ_LINGER,   &linger,   sizeof(linger));

    try { rep.bind(ZMQ_H_BIND); }
    catch (const zmq::error_t& e){ LOG("[ERR ] ZMQ H bind failed ", ZMQ_H_BIND, " ", e.what(), "\n"); return; }
    LOG("[HMAP] Listening for H on ", ZMQ_H_BIND, "\n");

    while (g_running.load()){
        zmq::message_t p1;
        auto ok1 = rep.recv(p1, zmq::recv_flags::none);
        if (!ok1){ continue; }

        int more = 0; size_t moresz = sizeof(more);
        rep.getsockopt(ZMQ_RCVMORE, &more, &moresz);

        std::string tag(static_cast<const char*>(p1.data()), p1.size());
        if (tag == "H" && more){
            zmq::message_t p2;
            auto ok2 = rep.recv(p2, zmq::recv_flags::none);
            if (!ok2){
                zmq::message_t reply(3); std::memcpy(reply.data(), "ERR", 3); rep.send(reply, zmq::send_flags::none); continue;
            }
            if (p2.size() != 9*sizeof(double)){
                zmq::message_t reply(3); std::memcpy(reply.data(), "BAD", 3); rep.send(reply, zmq::send_flags::none); continue;
            }
            {
                std::lock_guard<std::mutex> lk(g_h_mtx);
                std::memcpy(g_H, p2.data(), 9*sizeof(double));
                precompute_h_map_unlocked();
            }
            zmq::message_t reply(2); std::memcpy(reply.data(), "OK", 2); rep.send(reply, zmq::send_flags::none);
        } else if (!more){
            // Single-part control messages: "IDENTITY" to clear mapping
            if (tag == "IDENTITY"){
                std::lock_guard<std::mutex> lk(g_h_mtx);
                for (int k=0;k<9;++k) g_H[k] = (k%4==0)?1.0:0.0;
                g_h_src_idx.clear();
                g_h_ready.store(false);
                zmq::message_t reply(2); std::memcpy(reply.data(), "OK", 2); rep.send(reply, zmq::send_flags::none);
            } else {
                zmq::message_t reply(3); std::memcpy(reply.data(), "ERR", 3); rep.send(reply, zmq::send_flags::none);
            }
        } else {
            // Drain any remaining parts
            while (rep.getsockopt(ZMQ_RCVMORE, &more, &moresz), more){
                zmq::message_t d;
                (void)rep.recv(d, zmq::recv_flags::none);
            }
            zmq::message_t reply(3); std::memcpy(reply.data(), "ERR", 3); rep.send(reply, zmq::send_flags::none);
        }
    }

    rep.close();
    ctx.close();
}

static void camera_thread_func(){
    gpiod_line* line = request_edge_line(CAM_TRIG_CHIP, CAM_TRIG_LINE, CAM_EDGE, "cam");
    if (!line){ LOG("[CAM ] failed to arm\n"); return; }
    LOG("[CAM ] armed on ", CAM_TRIG_CHIP, ":", CAM_TRIG_LINE, " edge=", edge_name(CAM_EDGE), "\n");

    while (g_running.load()){
        timespec to{0, 500*1000*1000};
        int rv = gpiod_line_event_wait(line, &to);
        if (rv < 0){ LOG("[CAM ] event_wait error\n"); break; }
        if (rv == 0){ continue; }

        gpiod_line_event ev;
        if (gpiod_line_event_read(line, &ev) < 0){ LOG("[CAM ] event_read error\n"); continue; }

        uint64_t idx = cam_frame_idx.fetch_add(1) + 1;
        camera_trigger_count.fetch_add(1);
        last_cam_idx.store(idx);
        int64_t  tns = (int64_t)ev.ts.tv_sec*1000000000LL + ev.ts.tv_nsec;

        // Promote through L-frame FIFO (age on camera trigger)
        int cur = latest_mask_id.load();
        int promoted = -1;
        {
            std::lock_guard<std::mutex> lk(cam_fifo_mtx);
            cam_fifo.push_back(cur);
            if ((int)cam_fifo.size() > LATENCY_FRAMES){
                promoted = cam_fifo.front();
                cam_fifo.pop_front();
            }
        }
        if (promoted >= 0){
            g_ready_q.push(promoted); // <-- queue, not single-slot
        }

        // ---- Mapping: we will map camera frame to the last projector event <= (ts_adj+eps)
        int saved_mask = -1;
        uint64_t matched_pidx = 0;
        {
            const int64_t shift_ns = CAM_TS_OFFSET_US * 1000LL;
            const int64_t eps_ns   = MAP_EPS_US * 1000LL;
            const int64_t ts_adj   = tns + shift_ns;

            std::lock_guard<std::mutex> lk(proj_hist_mtx);
            for (auto it = proj_hist.rbegin(); it != proj_hist.rend(); ++it){
                if (it->t_ns <= ts_adj + eps_ns){
                    saved_mask = it->mask_id;    // <-- this is the *visible* mask for that projector frame
                    matched_pidx = it->pidx;
                    break;
                }
            }
        }
        last_matched_proj_for_cam.store(matched_pidx);

        static std::mutex csv_mtx;
        static std::ofstream csv;
        if (!csv.is_open()){
            csv.open(MAP_CSV_PATH.c_str(), std::ios::out | std::ios::trunc);
            if (!csv.is_open()){
                LOG("[ERR ] cannot open ", MAP_CSV_PATH, " for writing\n");
            } else {
                LOG("[MAP ] writing to ", MAP_CSV_PATH, "\n");
            }
        }
        if (csv.is_open()){
            if (saved_mask >= 0){
                std::lock_guard<std::mutex> lk(csv_mtx);
                long long out_cam_idx = (long long)idx - CAM_WARMUP-1;
                if (out_cam_idx >= 1){
                    csv << saved_mask << "," << out_cam_idx << "\n"; // start at 1 after warm-up
                } else {
                    csv << saved_mask << "," << out_cam_idx << "\n"; // negative/zero warm-up rows
                }
                csv.flush();
            }
        }

        int vis_id = last_visible_mask_id.load();
        uint64_t vis_proj = last_visible_proj_idx.load();
        LOG("[CAM ] frame #", idx, " @", tns, " ns -> PROJ #", vis_proj, " visible_id=", vis_id,
            " (mapped mask=", saved_mask, ")\n");
    }
    gpiod_line_release(line);
}

static void projector_thread_func(){
    gpiod_line* line = request_edge_line(PROJ_TRIG_CHIP, PROJ_TRIG_LINE, PROJ_EDGE, "proj");
    if (!line){ LOG("[PROJ] failed to arm\n"); return; }
    LOG("[PROJ] armed at ", now_ns(), " ns on ", PROJ_TRIG_CHIP, ":", PROJ_TRIG_LINE, " edge=", edge_name(PROJ_EDGE), "\n");

    int last_vis = -1;

    // Status publisher (PUB) for visible id per projector trigger
    zmq::context_t pub_ctx(1);
    zmq::socket_t  pub_sock(pub_ctx, ZMQ_PUB);
    int linger   = 0;   pub_sock.setsockopt(ZMQ_LINGER,   &linger,   sizeof(linger));
    try { pub_sock.bind("tcp://*:5562"); }
    catch (const zmq::error_t& e){ LOG("[PROJ] PUB bind failed tcp://*:5562 ", e.what(), "\n"); }

    while (g_running.load()){
        timespec to{0, 500*1000*1000};
        int rv = gpiod_line_event_wait(line, &to);
        if (rv < 0){ LOG("[PROJ] event_wait error\n"); break; }
        if (rv == 0){ continue; }

        gpiod_line_event ev;
        if (gpiod_line_event_read(line, &ev) < 0){ LOG("[PROJ] event_read error\n"); continue; }

        uint64_t pidx = proj_trig_idx.fetch_add(1) + 1;
        int64_t tns = (int64_t)ev.ts.tv_sec*1000000000LL + ev.ts.tv_nsec;

        // 1) Determine which mask is *visible* on THIS projector frame.
        //    Pop one id from the "swapped" queue if available; else reuse last.
        int vis_id;
        int popped = -1;
        if (g_swapped_q.try_pop(popped)) {
            vis_id = popped;
        } else {
            vis_id = last_vis; // no new swap since last vblank -> same content visible
        }
        last_vis = vis_id;
        last_visible_mask_id.store(vis_id);
        last_visible_proj_idx.store(pidx);

        // Publish status for GUI pacing (best-effort)
        try {
            std::ostringstream oss; oss << "{\"pidx\":" << pidx << ",\"vis_id\":" << vis_id << "}";
            auto s = oss.str();
            zmq::message_t m(s.size());
            std::memcpy(m.data(), s.data(), s.size());
            pub_sock.send(m, zmq::send_flags::dontwait);
        } catch(...) {}

        {
            std::lock_guard<std::mutex> lk(proj_hist_mtx);
            proj_hist.push_back({pidx, tns, vis_id});
            if (proj_hist.size() > PROJ_HIST_MAX) proj_hist.pop_front();
        }

        // 2) Schedule what to DRAW for the *next* projector frame: pop from ready queue
        int next_id = -1;
        if (g_ready_q.try_pop(next_id)){
            pending_draw_proj_idx.store(pidx);      // will become visible at pidx+1
            pending_draw_id.store(next_id);
            if (g_win) glfwPostEmptyEvent();
            LOG("[PROJ] trig #", pidx, " @", tns, " ns -> visible_id=", vis_id,
                " | queued next_id=", next_id, " (readyQ=", g_ready_q.size(), ")\n");
        } else {
            LOG("[PROJ] trig #", pidx, " @", tns, " ns -> visible_id=", vis_id,
                " | (no ready id; L=", LATENCY_FRAMES, ")\n");
        }
    }
    gpiod_line_release(line);
}

// ---------- robust CLI parsing ----------
static void parse_cli(int argc, char** argv){
    for (int i=1;i<argc;++i){
        std::string a(argv[i]);
        auto starts = [&](const char* p){ return a.rfind(p, 0) == 0; };

        if      (starts("--proj-chip="))   PROJ_TRIG_CHIP = trim(a.substr(12));
        else if (starts("--proj-line="))   PROJ_TRIG_LINE = safe_stoi(a.substr(12), PROJ_TRIG_LINE, "proj-line");
        else if (starts("--proj-edge=")) { auto v = trim(a.substr(12));
            PROJ_EDGE = (v=="rising")?Edge::Rising:(v=="falling")?Edge::Falling:Edge::Both; }

        else if (starts("--cam-chip="))    CAM_TRIG_CHIP  = trim(a.substr(11));
        else if (starts("--cam-line="))    CAM_TRIG_LINE  = safe_stoi(a.substr(11), CAM_TRIG_LINE, "cam-line");
        else if (starts("--cam-edge="))  { auto v = trim(a.substr(11));
            CAM_EDGE = (v=="rising")?Edge::Rising:(v=="falling")?Edge::Falling:Edge::Both; }

        else if (starts("--latency-frames=")) LATENCY_FRAMES = std::max(0, safe_stoi(a.substr(18), LATENCY_FRAMES, "latency-frames"));
        else if (starts("--bind="))           ZMQ_BIND       = trim(a.substr(7));
        else if (starts("--swap-interval="))  SWAP_INTERVAL  = safe_stoi(a.substr(16), SWAP_INTERVAL, "swap-interval");
        else if (starts("--monitor-index="))  MONITOR_PICK   = safe_stoi(a.substr(16), MONITOR_PICK, "monitor-index");

        else if (starts("--visible-id="))     VISIBLE_ID     = safe_stoi(a.substr(13), VISIBLE_ID?1:0, "visible-id") != 0;
        else if (a=="--visible-id")           VISIBLE_ID     = true;

        else if (starts("--overlay-style=")) {
            auto v = trim(a.substr(16));
            OVERLAY_STYLE = (v=="digits") ? 1 : 0;
        }
        else if (starts("--overlay-cell="))   OVERLAY_CELL   = std::max(4, safe_stoi(a.substr(15), OVERLAY_CELL, "overlay-cell"));
        else if (starts("--overlay-pos="))    parse_pos_pair(a.substr(14), OVERLAY_OFF_X, OVERLAY_OFF_Y);
        else if (starts("--overlay-bg="))     OVERLAY_BG     = safe_stoi(a.substr(13), OVERLAY_BG?1:0, "overlay-bg") != 0;
        else if (starts("--overlay-bottom=")){
            auto v = trim(a.substr(17));
            if      (v=="mask") OVERLAY_BOTTOM_MODE = 0;
            else if (v=="proj") OVERLAY_BOTTOM_MODE = 1;
            else if (v=="none") OVERLAY_BOTTOM_MODE = 2;
        }

        else if (starts("--cam-ts-offset-us=")) CAM_TS_OFFSET_US = safe_stoll(a.substr(20), CAM_TS_OFFSET_US, "cam-ts-offset-us");
        else if (starts("--map-eps-us="))       MAP_EPS_US       = safe_stoll(a.substr(13), MAP_EPS_US, "map-eps-us");
        else if (starts("--map-csv="))          MAP_CSV_PATH     = trim(a.substr(10));
        else if (starts("--cam-warmup="))       CAM_WARMUP       = safe_stoi(a.substr(12), CAM_WARMUP, "cam-warmup");
        else if (starts("--h-bind="))           ZMQ_H_BIND       = trim(a.substr(9));
        else if (starts("--horiz-flip="))       HORIZ_FLIP       = safe_stoi(a.substr(13), HORIZ_FLIP, "horiz-flip");
        else if (starts("--h-file="))           H_FILE_PATH      = trim(a.substr(9));

        else if (a=="-h" || a=="--help"){
            std::cout <<
            "Usage: projector [options]\n"
            "  --proj-chip=/dev/gpiochipN\n"
            "  --proj-line=N\n"
            "  --proj-edge=rising|falling|both\n"
            "  --cam-chip=/dev/gpiochipN\n"
            "  --cam-line=N\n"
            "  --cam-edge=rising|falling|both\n"
            "  --latency-frames=L\n"
            "  --bind=tcp://*:5558\n"
            "  --swap-interval=0|1\n"
            "  --monitor-index=N\n"
            "  --visible-id[=0|1]\n"
            "  --overlay-style=digits|barcode\n"
            "  --overlay-cell=N\n"
            "  --overlay-pos=X,Y\n"
            "  --overlay-bg=0|1\n"
            "  --overlay-bottom=mask|proj|none\n"
            "  --cam-ts-offset-us=S     (default 0; negative if cam trigger is late)\n"
            "  --map-eps-us=E           (default 500)\n"
            "  --map-csv=path           (default ./mask_map.csv)\n"
            "  --cam-warmup=N           (default 10)\n";
            std::exit(0);
        }
    }
    // echo parsed config
    LOG("[CLI ] proj ", PROJ_TRIG_CHIP, ":", PROJ_TRIG_LINE, " ", edge_name(PROJ_EDGE),
        " , cam ", CAM_TRIG_CHIP, ":", CAM_TRIG_LINE, " ", edge_name(CAM_EDGE),
        " , L=", LATENCY_FRAMES,
        " , bind=", ZMQ_BIND,
        " , swap-interval=", SWAP_INTERVAL,
        " , monitor-index=", MONITOR_PICK,
        " , visible-id=", (VISIBLE_ID?1:0),
        " , overlay-style=", (OVERLAY_STYLE? "digits":"barcode"),
        " , overlay-cell=", OVERLAY_CELL,
        " , overlay-pos=", OVERLAY_OFF_X, ",", OVERLAY_OFF_Y,
        " , overlay-bg=", (OVERLAY_BG?1:0),
        " , overlay-bottom=", (OVERLAY_BOTTOM_MODE==0?"mask":OVERLAY_BOTTOM_MODE==1?"proj":"none"),
        " , cam-ts-offset-us=", CAM_TS_OFFSET_US,
        " , map-eps-us=", MAP_EPS_US,
        " , map-csv=", MAP_CSV_PATH,
        " , cam-warmup=", CAM_WARMUP,
        " , h-bind=", ZMQ_H_BIND,
        " , horiz-flip=", HORIZ_FLIP, "\n");
}

// ---------- monitor pick ----------
static GLFWmonitor* pick_monitor(){
    int count = 0;
    GLFWmonitor** mons = glfwGetMonitors(&count);
    if (!mons || count == 0) return nullptr;

    if (MONITOR_PICK >= 0 && MONITOR_PICK < count){
        return mons[MONITOR_PICK];
    }
    int bestX = -100000000, best = 0;
    for (int i=0;i<count;++i){
        int mx=0,my=0; glfwGetMonitorPos(mons[i], &mx, &my);
        const char* name = glfwGetMonitorName(mons[i]);
        LOG("Monitor ", i, " ", (name?name:"unknown"), " at +", mx, "+", my, "\n");
        if (mx > bestX){ bestX = mx; best = i; }
    }
    return mons[best];
}

// ---------- signal ----------
static void on_sig(int){ g_running.store(false); if (g_win) glfwPostEmptyEvent(); }

// ---------- main ----------
int main(int argc, char** argv){
    parse_cli(argc, argv);

    // If an on-disk H file is provided, preload it before threads/GL start
    if (!H_FILE_PATH.empty()){
        load_h_from_text_file(H_FILE_PATH);
    }

    // start background workers before GL
    std::thread th_zmq(zmq_thread_func);
    std::thread th_h(h_zmq_thread_func);
    std::thread th_cam(camera_thread_func);
    std::thread th_proj(projector_thread_func);

    // GLFW setup and window
    if (!glfwInit()){ std::cerr << "GLFW init failed\n"; g_running.store(false); th_proj.join(); th_cam.join(); th_zmq.join(); return 1; }
    std::signal(SIGINT,  on_sig);
    std::signal(SIGTERM, on_sig);

    GLFWmonitor* proj = pick_monitor();
    if (!proj){ std::cerr << "No monitor found\n"; g_running.store(false); th_proj.join(); th_cam.join(); th_zmq.join(); glfwTerminate(); return 1; }

    int mx=0,my=0; glfwGetMonitorPos(proj, &mx, &my);
    const GLFWvidmode* mode = glfwGetVideoMode(proj);
    if (!mode){ std::cerr << "No video mode\n"; g_running.store(false); th_proj.join(); th_cam.join(); th_zmq.join(); glfwTerminate(); return 1; }
    LOG("Using monitor at +", mx, "+", my, " (", mode->width, "x", mode->height, "@", mode->refreshRate, "Hz)\n");

    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);

    g_win = glfwCreateWindow(mode->width, mode->height, "Mask Projection", nullptr, nullptr);
    if (!g_win){ std::cerr << "Window creation failed\n"; g_running.store(false); th_proj.join(); th_cam.join(); th_zmq.join(); glfwTerminate(); return 1; }
    glfwSetWindowPos(g_win, mx, my);
    glfwShowWindow(g_win);
    glfwMakeContextCurrent(g_win);
    glfwSwapInterval(SWAP_INTERVAL);

    // warm up with black so WM maps the window (FULLSCREEN content, no decorations)
    {
        std::vector<unsigned char> black(WIDTH * HEIGHT, 0);
        for (int i=0;i<2;++i){
            draw_mask_pixels(black.data(), WIDTH, HEIGHT);
            glfwSwapBuffers(g_win);
            glfwPollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    const size_t expected_bytes = size_t(WIDTH) * size_t(HEIGHT);

    // Main loop, render when projector thread posts a pending id
    while (g_running.load() && !glfwWindowShouldClose(g_win)){
        glfwWaitEventsTimeout(0.1);

        if (glfwGetWindowAttrib(g_win, GLFW_ICONIFIED)){
            glfwRestoreWindow(g_win);
            glfwSetWindowPos(g_win, mx, my);
        }

        int id = pending_draw_id.exchange(-1);
        if (id >= 0){
            const unsigned char* ptr = nullptr; size_t n = 0;
            if (g_cache.get(id, ptr, n) && n == expected_bytes){
                auto t_before = now_ns();
                // Apply homography mapping if available
                static std::vector<unsigned char> warped;
                bool use_h = g_h_ready.load();
                if (use_h){
                    warp_mask_nn(ptr, warped);
                }

                // Prepare overlay buffers; draw timing depends on use_h
                int ov_w = 0, ov_h = 0;
                static std::vector<unsigned char> ov;
                bool overlay_built = false;

                if (VISIBLE_ID){
                    // top row: running counter starting at 1
                    uint64_t ctr = draw_counter.fetch_add(1) + 1;

                    // bottom row: per setting
                    std::string bottom;
                    if (OVERLAY_BOTTOM_MODE == 0) {
                        // Show mapping label with clear spacing: mask_id    cam_idx    proj_idx
                        // Note: draw_number_row only renders digits and spaces; separators like ':' or '@' are not drawn.
                        long long cam_idx = (long long)last_cam_idx.load() - CAM_WARMUP;
                        uint64_t proj_idx = last_matched_proj_for_cam.load();
                        bottom = std::to_string(std::max(0, id)) + "    " + std::to_string(cam_idx) + "    " + std::to_string(proj_idx);
                    } else if (OVERLAY_BOTTOM_MODE == 1) {
                        bottom = std::to_string(pending_draw_proj_idx.load()); // target projector idx
                    } else {
                        bottom = "";
                    }

                    if (OVERLAY_STYLE == 1){
                        build_overlay_digits(ctr, bottom, OVERLAY_CELL, ov, ov_w, ov_h);
                    } else {
                        uint64_t pidx_full = pending_draw_proj_idx.load();
                        uint8_t id8 = uint8_t(std::max(0, id) & 0xFF);
                        uint8_t p8  = uint8_t(pidx_full & 0xFF);
                        uint8_t hb8 = uint8_t((pidx_full >> 8) & 0xFF);
                        build_overlay_barcode(id8, p8, hb8, OVERLAY_CELL, ov, ov_w, ov_h);
                    }

                    if (use_h){
                        // Prewarp overlay using same mapping
                        static std::vector<unsigned char> ov_full;  // source-space full overlay
                        static std::vector<unsigned char> ov_warped; // dest-space overlay
                        static std::vector<unsigned char> plate_src; // source-space plate mask
                        static std::vector<unsigned char> plate_w;   // dest-space plate mask

                        ov_full.assign((size_t)WIDTH * (size_t)HEIGHT, 0);
                        blit_onto_fullscreen(ov_full, WIDTH, HEIGHT, ov, ov_w, ov_h, OVERLAY_OFF_X, OVERLAY_OFF_Y);

                        if (OVERLAY_BG){
                            plate_src.assign((size_t)WIDTH * (size_t)HEIGHT, 0);
                            // draw solid rect in source space where overlay sits
                            int x0 = OVERLAY_OFF_X, y0 = OVERLAY_OFF_Y;
                            for (int y = 0; y < ov_h; ++y){
                                int dy = y0 + y; if (dy < 0 || dy >= HEIGHT) continue;
                                int dx0 = x0; int run = ov_w;
                                if (dx0 < 0){ run += dx0; dx0 = 0; }
                                if (dx0 + run > WIDTH){ run = WIDTH - dx0; }
                                if (run <= 0) continue;
                                unsigned char* dst = plate_src.data() + (size_t)dy * (size_t)WIDTH + (size_t)dx0;
                                std::memset(dst, 255, (size_t)run);
                            }
                            warp_mask_nn(plate_src.data(), plate_w);
                        } else {
                            plate_w.clear();
                        }

                        warp_mask_nn(ov_full.data(), ov_warped);

                        // Apply black plate in dest space
                        if (!plate_w.empty()){
                            const size_t N = (size_t)WIDTH * (size_t)HEIGHT;
                            for (size_t i = 0; i < N; ++i){
                                if (plate_w[i]) warped[i] = 0;
                            }
                        }
                        // Composite overlay digits on top (max blend)
                        {
                            const size_t N = (size_t)WIDTH * (size_t)HEIGHT;
                            for (size_t i = 0; i < N; ++i){
                                unsigned char v = ov_warped[i];
                                if (v > warped[i]) warped[i] = v;
                            }
                        }
                    } else {
                        overlay_built = true; // defer GL overlay until after base mask draw
                    }
                }

                if (use_h){
                    draw_mask_pixels(warped.data(), WIDTH, HEIGHT);
                } else {
                    draw_mask_pixels(ptr, WIDTH, HEIGHT);
                    if (VISIBLE_ID && overlay_built){
                        draw_overlay_pixels(ov.data(), ov_w, ov_h, OVERLAY_OFF_X, OVERLAY_OFF_Y);
                    }
                }
                glfwSwapBuffers(g_win);
                auto t_after = now_ns();

                // Tell projector thread which id actually swapped (will be visible on the next projector trigger)
                g_swapped_q.push(id);

                LOG("[DRAW] id=", id, " target_pidx+1=", pending_draw_proj_idx.load()+1,
                    " draw+swap=", (t_after - t_before)/1000000.0, " ms, swappedQ=", g_swapped_q.size(), "\n");
            } else {
                static std::vector<unsigned char> black(WIDTH*HEIGHT, 0);
                draw_mask_pixels(black.data(), WIDTH, HEIGHT);
                glfwSwapBuffers(g_win);
                g_swapped_q.push(-1);
                LOG("[DRAW] id=", id, " not cached, drew black\n");
            }
        }
    }

    // shutdown
    g_running.store(false);
    glfwDestroyWindow(g_win);
    glfwTerminate();

    th_proj.join();
    th_cam.join();
    th_zmq.join();
    th_h.join();

    LOG("Bye.\n");
    return 0;
}
