#include <zmq.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

static int W = 1920;
static int H = 1080;

static const char* ENDPOINT_DEFAULT = "tcp://127.0.0.1:5558";

static void usage(const char* prog){
    std::fprintf(stderr,
        "Usage: %s [--endpoint=tcp://127.0.0.1:5558] [--fps=60] [--w=1920] [--h=1080] [--rect-w=120] [--rect-h=120] [--speed=8]\n",
        prog);
}

static int parse_int(const std::string& s, int defv){
    try{ return std::stoi(s); } catch(...){ return defv; }
}

int main(int argc, char** argv){
    std::string endpoint = ENDPOINT_DEFAULT;
    int fps = 60;
    int speed_px = 8; // pixels per frame

    for (int i=1;i<argc;++i){
        std::string a(argv[i]);
        auto starts = [&](const char* p){ return a.rfind(p, 0) == 0; };
        if      (starts("--endpoint=")) endpoint = a.substr(11);
        else if (starts("--fps="))      fps      = parse_int(a.substr(6), fps);
        else if (starts("--w="))        W        = parse_int(a.substr(4), W);
        else if (starts("--h="))        H        = parse_int(a.substr(4), H);
        else if (starts("--speed="))    speed_px = parse_int(a.substr(8), speed_px);
        else if (a=="-h" || a=="--help"){ usage(argv[0]); return 0; }
    }

    // Environment overrides for size
    const char* envw = std::getenv("MASK_W");
    const char* envh = std::getenv("MASK_H");
    if (envw) { int v = std::atoi(envw); if (v>0) W=v; }
    if (envh) { int v = std::atoi(envh); if (v>0) H=v; }

    if (fps <= 0)    fps = 60;
    if (speed_px <= 0) speed_px = 8;

    void* ctx = zmq_ctx_new();
    if (!ctx){ std::perror("zmq_ctx_new"); return 1; }
    void* s = zmq_socket(ctx, ZMQ_PUSH);
    if (!s){ std::perror("zmq_socket"); zmq_ctx_term(ctx); return 1; }
    int hwm = 4; zmq_setsockopt(s, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    int immediate = 1; zmq_setsockopt(s, ZMQ_IMMEDIATE, &immediate, sizeof(immediate));
    int sndtimeo = 0; zmq_setsockopt(s, ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
    if (zmq_connect(s, endpoint.c_str()) != 0){ std::perror("zmq_connect"); zmq_close(s); zmq_ctx_term(ctx); return 1; }

    std::vector<uint8_t> img((size_t)W*(size_t)H, 0);
    using clock_t = std::chrono::steady_clock;
    using dur_t   = clock_t::duration;
    const dur_t interval = std::chrono::duration_cast<dur_t>(std::chrono::duration<double>(1.0 / (double)fps));
    int id = 0;
    int x = 0;
    const int scale = 12; // pixel scale for 5x7 font (larger)
    const char* text = "STIMscope";
    const int glyph_w = 5 * scale;
    const int glyph_h = 7 * scale;
    const int spacing = scale; // inter-glyph space
    const int text_w = (int)std::strlen(text) * (glyph_w + spacing) - spacing;
    const int y = (H - glyph_h)/2;

    // 5x7 font for needed chars
    struct Glyph { char c; const char* rows[7]; };
    static const Glyph font[] = {
        {'S', {"01110","10000","11100","00010","00010","10010","01100"}},
        {'T', {"11111","00100","00100","00100","00100","00100","00100"}},
        {'I', {"11111","00100","00100","00100","00100","00100","11111"}},
        {'M', {"10001","11011","10101","10101","10001","10001","10001"}},
        {'C', {"01110","10001","10000","10000","10000","10001","01110"}},
        {'O', {"01110","10001","10001","10001","10001","10001","01110"}},
        {'P', {"11110","10001","10001","11110","10000","10000","10000"}},
        {'E', {"11111","10000","11110","10000","10000","10000","11111"}},
        // lowercase approximations
        {'s', {"00000","01110","10000","01100","00010","11100","00000"}},
        {'c', {"00000","01110","10000","10000","10000","01110","00000"}},
        {'o', {"00000","01110","10001","10001","10001","01110","00000"}},
        {'p', {"00000","11110","10001","10001","11110","10000","10000"}},
        {'e', {"00000","01110","10001","11111","10000","01110","00000"}},
    };
    auto find_glyph = [&](char ch)->const Glyph*{
        for (const auto& g : font) if (g.c == ch) return &g;
        // fallback to 'O'
        for (const auto& g : font) if (g.c == 'O') return &g;
        return nullptr;
    };

    auto t_next = clock_t::now();
    std::fprintf(stdout, "Streaming moving rectangle at %d fps to %s (%dx%d)\n", fps, endpoint.c_str(), W, H);
    std::fflush(stdout);

    while (true){
        // Build frame
        std::memset(img.data(), 0, img.size());
        // Draw text string at x,y using 5x7 scaled font
        int pen_x = x;
        for (const char* p = text; *p; ++p){
            const Glyph* g = find_glyph(*p);
            if (!g){ pen_x += glyph_w + spacing; continue; }
            for (int ry = 0; ry < 7; ++ry){
                const char* rowbits = g->rows[ry];
                for (int rx = 0; rx < 5; ++rx){
                    if (rowbits[rx] == '1'){
                        // draw scaled block
                        int px0 = pen_x + rx*scale;
                        int py0 = y + ry*scale;
                        for (int sy = 0; sy < scale; ++sy){
                            int yy = py0 + sy; if (yy < 0 || yy >= H) continue;
                            uint8_t* row = img.data() + (size_t)(H - 1 - yy) * (size_t)W;
                            for (int sx = 0; sx < scale; ++sx){
                                int xx = px0 + sx; if (xx < 0 || xx >= W) continue;
                                row[xx] = 255;
                            }
                        }
                    }
                }
            }
            pen_x += glyph_w + spacing;
        }

        // Advance position
        x += speed_px;
        if (x > W) x = -text_w; // re-enter from left

        // Send multipart: meta, payload
        ++id;
        char meta[64];
        std::snprintf(meta, sizeof(meta), "{\"id\":%d}", id);
        zmq_msg_t m1; zmq_msg_init_size(&m1, std::strlen(meta));
        std::memcpy(zmq_msg_data(&m1), meta, std::strlen(meta));
        int rc = zmq_msg_send(&m1, s, ZMQ_SNDMORE);
        zmq_msg_close(&m1);
        if (rc < 0){ /* dropped */ }

        zmq_msg_t m2; zmq_msg_init_size(&m2, img.size());
        std::memcpy(zmq_msg_data(&m2), img.data(), img.size());
        rc = zmq_msg_send(&m2, s, 0);
        zmq_msg_close(&m2);

        // pace
        t_next += interval;
        auto now = clock_t::now();
        if (t_next > now) std::this_thread::sleep_until(t_next);
        else t_next = now;
    }

    // unreachable in normal use
    zmq_close(s);
    zmq_ctx_term(ctx);
    return 0;
}


