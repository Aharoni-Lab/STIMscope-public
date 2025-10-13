I found the projection engine in `ZMQ_sender_mask/main.cpp`, located the GL render loop, the ZMQ/CAM/PROJ threads, the queues, and the exact log emit points. I’ll walk through the data flow and timing, then decode your sample logs.

### What the projection engine does
- **Receives masks via ZMQ**: meta JSON + raw 8-bit 1920×1080 bytes; caches by `id`.
- **Ages by camera triggers**: an L-frame FIFO on camera edges decides when the latest ID becomes “ready” to draw.
- **On projector triggers**: records which mask is visible now; pops one ready ID to schedule for the next projector frame.
- **GL thread**: draws the scheduled ID and swaps at vsync, making it visible on the following projector trigger.
- **Overlays**: top row is a running counter; bottom row shows mapping “mask_id  cam_idx  proj_idx” per your request [[memory:8799416]].

### Threads and queues (and where logs come from)
- **ZMQ receiver**: caches masks; can enqueue immediately; toggles overlay at runtime.
```648:710:ZMQ_sender_mask/main.cpp
static void zmq_thread_func(){
    ...
    g_cache.put(id, static_cast<const unsigned char*>(part2.data()), part2.size());
    latest_mask_id.store(id);
    bool immediate = parse_flag_from_json(meta, "immediate");
    if (immediate){
        while (g_ready_q.size() > 4){ int drop = -1; g_ready_q.try_pop(drop); }
        g_ready_q.push(id);
    }
    int vis = parse_opt_bool_from_json(meta, "visible_id");
    if (vis >= 0){ VISIBLE_ID = (vis != 0); }
    LOG("[ZMQ ] received id=", id, immediate?" (immediate)": "", ", cached ", part2.size(), " bytes\n");
}
```

- **Camera trigger**: L-frame FIFO aging; maps camera frame to last projector event; writes CSV; logs “CAM … mapped mask=…”.
```795:856:ZMQ_sender_mask/main.cpp
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
    g_ready_q.push(promoted);
}
// Map cam ts to last projector <= ts+eps, then CSV and log
...
LOG("[CAM ] frame #", idx, " @", tns, " ns -> PROJ #", vis_proj, " visible_id=", vis_id,
    " (mapped mask=", saved_mask, ")\n");
```

- **Projector trigger**: determines `visible_id` for THIS trigger from swap queue; logs “PROJ trig … visible_id=…”. Schedules one next ID to draw for the next frame.
```886:925:ZMQ_sender_mask/main.cpp
// 1) Which mask is visible now
int vis_id;
int popped = -1;
if (g_swapped_q.try_pop(popped)) { vis_id = popped; } else { vis_id = last_vis; }
last_visible_mask_id.store(vis_id);
last_visible_proj_idx.store(pidx);
// 2) Schedule next draw
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
```

- **GL render loop**: draws `pending_draw_id`, swaps, pushes ID to `g_swapped_q`, logs draw time.
```1108:1221:ZMQ_sender_mask/main.cpp
int id = pending_draw_id.exchange(-1);
if (id >= 0){
    const unsigned char* ptr = nullptr; size_t n = 0;
    if (g_cache.get(id, ptr, n) && n == expected_bytes){
        auto t_before = now_ns();
        bool use_h = g_h_ready.load();
        if (use_h){ ... warp_mask_* on CPU over WIDTH*HEIGHT ... }
        if (VISIBLE_ID){ ... build digits/barcode and compose (CPU or GL) ... }
        if (use_h){ draw_mask_pixels(warped.data(), WIDTH, HEIGHT); }
        else {
            draw_mask_pixels(ptr, WIDTH, HEIGHT);
            if (VISIBLE_ID && overlay_built){
                draw_overlay_pixels(ov.data(), ov_w, ov_h, OVERLAY_OFF_X, OVERLAY_OFF_Y);
            }
        }
        glfwSwapBuffers(g_win);
        auto t_after = now_ns();
        g_swapped_q.push(id);
        LOG("[DRAW] id=", id, " target_pidx+1=", pending_draw_proj_idx.load()+1,
            " draw+swap=", (t_after - t_before)/1000000.0, " ms, swappedQ=", g_swapped_q.size(), "\n");
    }
}
```

- **Queues** used for handoff:
```111:136:ZMQ_sender_mask/main.cpp
struct IntQueue { ... };    // lock-protected FIFO
static IntQueue g_ready_q;   // camera-aged IDs to draw next
static IntQueue g_swapped_q; // IDs that actually swapped (visible on THIS projector frame)
```

### How OpenGL is used and what’s modified on the GPU
- Uses legacy immediate-mode `glDrawPixels` to blit the 8-bit luminance image into the back buffer; then `glfwSwapBuffers` at vsync.
```196:206:ZMQ_sender_mask/main.cpp
static void draw_mask_pixels(const void* data, int w, int h){
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_DITHER);
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelZoom(1.0f, -1.0f);      // flip vertical
    glRasterPos2f(-1.f, 1.f);      // top-left origin
    glDrawPixels(w, h, GL_LUMINANCE, GL_UNSIGNED_BYTE, data);
}
```
- Overlays: either composed on CPU (when H is applied) or drawn via another `glDrawPixels` in an ortho pass. No textures/shaders; it only alters the back buffer and simple matrix state.
```450:481:ZMQ_sender_mask/main.cpp
glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity();
glOrtho(0, winW, 0, winH, -1, 1);
glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
...
glRasterPos2i(x, y);
glDrawPixels(ow, oh, GL_LUMINANCE, GL_UNSIGNED_BYTE, px);
```
- Vsync (line up with projector refresh):
```1083:1085:ZMQ_sender_mask/main.cpp
glfwMakeContextCurrent(g_win);
glfwSwapInterval(SWAP_INTERVAL);
```

### Flow of information (end-to-end)
- Mask produced → [meta JSON, 2,073,600 bytes] sent over ZMQ PULL.
- ZMQ thread caches under `id`; may enqueue to `g_ready_q`; logs “[ZMQ] received id=…”.
- Camera trigger → advances FIFO; pushes “promoted” to `g_ready_q`; maps to last projector event and writes CSV; logs “[CAM] … (mapped mask=…)”.
- Projector trigger → pops a `swapped` ID to get the visible `vis_id`; logs “[PROJ] trig … visible_id=…”. Also pops one from `g_ready_q` to post `pending_draw_id` for the next frame; logs “queued next_id=…”.
- GL thread → draws `pending_draw_id`, swaps at vsync, pushes the ID into `g_swapped_q`, logs “[DRAW] …”.

### When does a mask become visible?
- Scheduling point: at projector trigger #p, it sets `pending_draw_proj_idx=p`, targeting visibility at #p+1.
- If draw finishes after the next vblank, it slips to #p+2, #p+3, …
- Your logs show draw+swap ≈ 62–65 ms. At 60 Hz (16.7 ms), that’s ≈ 3–4 refreshes; so a newly scheduled mask typically lands ~4 triggers later.
- With L=1 camera FIFO, add ~one camera period before it’s eligible (e.g., ~33 ms at 30 Hz).

### Why is draw+swap ~62 ms?
- Includes CPU homography warp over all pixels, overlay prep, `glDrawPixels` upload (often synchronous on Jetson), and vsync wait. Net effect in your traces is ~62–65 ms, i.e., effective ~16 FPS content updates; about one visible update every ~4 projector triggers.
- Since `pending_draw_id` is a single slot, if the GL cannot keep up, intermediate scheduled IDs get overwritten.

### Decode of your sample log lines
- “[ZMQ] received id=3403, cached 2073600 bytes” → full 1920×1080 mask cached under id 3403.
- “[PROJ] trig #3520 … visible_id=3394 | queued next_id=3402 (readyQ=0)” → at trigger 3520, 3394 is visible; 3402 scheduled for the next frame.
- “[DRAW] id=3402 target_pidx+1=3525 draw+swap=62.57 ms” → GL drew 3402; target visibility 3525; total draw+swap ~62.57 ms.
- “trig #3525 … visible_id=3402” → confirms 3402 visible at projector trigger 3525.
- “[CAM] frame #1785 … -> PROJ #3525 visible_id=3402 (mapped mask=3402)” → camera frame mapped to that projector trigger and mask.

### Quick references
- GL draw path:
```196:206:ZMQ_sender_mask/main.cpp
glDrawPixels(w, h, GL_LUMINANCE, GL_UNSIGNED_BYTE, data);
```
- Log of draw timing:
```1220:1221:ZMQ_sender_mask/main.cpp
LOG("[DRAW] id=", id, " target_pidx+1=", ..., " draw+swap=", ..., " ms, swappedQ=", ...);
```
- Projector scheduling + log:
```914:922:ZMQ_sender_mask/main.cpp
pending_draw_proj_idx.store(pidx); pending_draw_id.store(next_id);
LOG("[PROJ] trig #", pidx, " ... | queued next_id=", next_id, " (readyQ=", g_ready_q.size(), ")\n");
```

- Vsync enable:
```1083:1085:ZMQ_sender_mask/main.cpp
glfwMakeContextCurrent(g_win); glfwSwapInterval(SWAP_INTERVAL);
```

- ZMQ receive + cache + log:
```692:710:ZMQ_sender_mask/main.cpp
g_cache.put(id, ..., part2.size());
LOG("[ZMQ ] received id=", id, ..., ", cached ", part2.size(), " bytes\n");
```

- Camera FIFO + mapping + log:
```806:855:ZMQ_sender_mask/main.cpp
if (promoted >= 0){ g_ready_q.push(promoted); }
LOG("[CAM ] frame #", idx, " ... visible_id=", vis_id, " (mapped mask=", saved_mask, ")\n");
```

- “Visible on THIS trigger” selection:
```886:896:ZMQ_sender_mask/main.cpp
if (g_swapped_q.try_pop(popped)) { vis_id = popped; } else { vis_id = last_vis; }
```

- Overlay digits builder (bottom “mask_id  cam_idx  proj_idx”):
```1130:1138:ZMQ_sender_mask/main.cpp
std::string bottom = std::to_string(std::max(0, id)) + "    " + std::to_string(cam_idx) + "    " + std::to_string(proj_idx);
```

- Main setup with monitor refresh rate:
```1069:1072:ZMQ_sender_mask/main.cpp
const GLFWvidmode* mode = glfwGetVideoMode(proj);
LOG("Using monitor ... (", mode->width, "x", mode->height, "@", mode->refreshRate, "Hz)\n");
```

- Single-slot handoff (why intermediate IDs are overwritten if GL lags):
```150:153:ZMQ_sender_mask/main.cpp
static std::atomic<int>      pending_draw_id{-1};
static std::atomic<uint64_t> pending_draw_proj_idx{0};
```

- Queues:
```133:136:ZMQ_sender_mask/main.cpp
static IntQueue g_ready_q; static IntQueue g_swapped_q;
```

- Homography warp (CPU):
```286:319:ZMQ_sender_mask/main.cpp
static void warp_mask_bilinear(const unsigned char* src, std::vector<unsigned char>& dst){ ... }
```

- Overlay draw:
```450:486:ZMQ_sender_mask/main.cpp
static void draw_overlay_pixels(...){ ... glDrawPixels(ow, oh, GL_LUMINANCE, GL_UNSIGNED_BYTE, px); ... }
```

- GL main loop target:
```1220:1221:ZMQ_sender_mask/main.cpp
" target_pidx+1=", pending_draw_proj_idx.load()+1
```

- Vsync gating:
```1084:1084:ZMQ_sender_mask/main.cpp
glfwSwapInterval(SWAP_INTERVAL);
```

- GPIO projector trigger log:
```919:924:ZMQ_sender_mask/main.cpp
LOG("[PROJ] trig #", pidx, " @", tns, " ns -> visible_id=", vis_id, ...);
```

- CSV mapping file:
```839:849:ZMQ_sender_mask/main.cpp
csv << saved_mask << "," << out_cam_idx << "\n";
```

- Overlay option echo:
```1003:1015:ZMQ_sender_mask/main.cpp
LOG("[CLI ] ... , overlay-bottom=", (OVERLAY_BOTTOM_MODE==0?"mask":OVERLAY_BOTTOM_MODE==1?"proj":"none"), ...);
```

- Homography status/log:
```268:270:ZMQ_sender_mask/main.cpp
g_h_ready.store(true); LOG("[HMAP] precomputed mapping (", W, "x", Ht, ")\n");
```

- Initialization and GL context:
```1079:1085:ZMQ_sender_mask/main.cpp
g_win = glfwCreateWindow(...); glfwMakeContextCurrent(g_win); glfwSwapInterval(SWAP_INTERVAL);
```

- Shutdown joins:
```1237:1241:ZMQ_sender_mask/main.cpp
th_proj.join(); th_cam.join(); th_zmq.join(); th_h.join();
```

- Your sample lines align exactly to these logs/paths; the ~62–65 ms draw+swap times indicate effective ~16 FPS update cadence with ~4 triggers between visible updates.

- Status: Located the GL, ZMQ, CAM, and PROJ paths plus logging sites in `ZMQ_sender_mask/main.cpp` and mapped them to your sample logs. Provided a precise end‑to‑end timing and flow explanation, and highlighted why draw+swap ≈ 62 ms leads to one visible update per ~4 projector triggers.

- Key takeaways:
  - **GL uploads + CPU warp dominate timing**.
  - **Visibility is quantized to projector triggers** and typically occurs ~1–4 triggers after scheduling; here ~4.
  - **Single-slot `pending_draw_id`** means intermediate IDs can be overwritten if the renderer lags.



