#include "gpu.hpp"

#include "basalt/core/card_link.hpp"
#include "basalt/log.hpp"
#include "basalt/model/multi_gpu.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/program/options.hpp"

#include <cstdlib>
#include <string>
#include <cstdio>
#include <cuda_runtime.h>

namespace basalt::program::cli {

std::optional<int> select_gpus(CommonOptions& o, std::vector<GpuCard>& chosen_cards, const model::ModelAdapter* adapter) {
    namespace bp = basalt::program;
    bp::GpuSpec spec;
    bp::GpuMode mode = bp::GpuMode::One;
    std::string gerr;
    if (!bp::parse_gpu_spec(o.gpu, spec, gerr) || !bp::parse_gpu_mode(o.gpu_mode, mode, gerr)) {
        if (o.gpu_plan) {
            bp::GpuChoice bad;
            bad.error = gerr;
            std::printf("%s\n", bp::gpu_plan_json({}, bad).c_str());
        } else {
            basalt::log::error("basalt", "%s", gerr.c_str());
        }
        return 2;
    }
    bool known = false;
    const std::vector<bp::GpuCard> visible = bp::visible_gpus(known);
    bp::GpuChoice choice = bp::choose_gpus(spec, mode, o.gpu_share, visible, known);
    if (choice.error.empty() && adapter != nullptr) choice.error = bp::check_gpu_modes(choice, adapter->model_name(), adapter->gpu_modes(), visible);
    bp::choose_vision_card(choice, o.vision_gpu, visible.size(), known);   // argus fork
    if (o.gpu_plan) {
        std::string lossy;
        for (const auto& f : o.lossy) lossy += (lossy.empty() ? "" : ",") + f.flag;
        std::printf("%s\n", bp::gpu_plan_json(visible, choice, o.arch, o.quality_flags, o.quality_window, o.plan_spec, o.max_context,
                                             o.model_name, lossy, o.community_kv).c_str());
        return choice.error.empty() ? 0 : 2;
    }
    if (!choice.error.empty()) { basalt::log::error("basalt", "%s", choice.error.c_str()); return 2; }
    for (const std::string& s : choice.notes) basalt::log::info("basalt", "%s", s.c_str());
    if (!choice.warning.empty()) basalt::log::warn("basalt", "%s", choice.warning.c_str());
    const std::vector<int> narrowed = bp::with_vision_card(choice);
    bp::apply_gpus(visible, narrowed);
    int seen = 0;
    if (cudaGetDeviceCount(&seen) != cudaSuccess || seen < (int) narrowed.size()) {
        basalt::log::error("basalt", "gpu %s: CUDA sees %d of the %zu GPUs chosen", o.gpu.empty() ? "0" : o.gpu.c_str(),
                           seen, narrowed.size());
        return 2;
    }
    o.vision_device = bp::vision_device(choice);
    if (choice.vision >= 0) {
        const bp::GpuCard vc = (size_t) choice.vision < visible.size() ? visible[(size_t) choice.vision] : bp::GpuCard{};
        basalt::log::info("basalt", "vision: %s%s image encoder (CUDA device %d)", vc.name.empty() ? "GPU" : vc.name.c_str(),
                          vc.smi >= 0 ? (" (nvidia-smi " + std::to_string(vc.smi) + ")").c_str() : "", o.vision_device);
    }
    // the cards the load receives: with several, each one's free memory before anything is loaded; the shares as given
    basalt::model::GpuCards& gc = o.gpus;
    gc = {};
    gc.mode = choice.mode;
    gc.given = choice.shares;
    gc.diag_cut = o.diag_gpu_cut;
    gc.diag_same = o.diag_gpu_same;
    for (size_t d = 0; d < choice.cards.size(); ++d) {
        const int i = choice.cards[d];
        basalt::model::GpuCards::Card c;
        c.device = (int) d;
        c.info = (size_t) i < visible.size() ? visible[(size_t) i] : bp::GpuCard{};
        c.index = i;
        c.label = bp::gpu_label(visible, i);
        if (choice.cards.size() > 1) c.free = basalt::core::card_free_bytes((int) d);   // (one card: no context made here)
        gc.cards.push_back(c);
        chosen_cards.push_back(c.info);
    }
    int drv = 0, rt = 0;
    (void) cudaDriverGetVersion(&drv);
    (void) cudaRuntimeGetVersion(&rt);
    basalt::log::info("basalt", "system: %s, CUDA driver %d.%d / runtime %d.%d",
                      basalt::platform::host_os_name().c_str(),
                      drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);
    basalt::log::info("basalt", "%s", basalt::model::gpu_start_line(gc).c_str());
    // a split hands a window's activations from one card to the next and a peer reads its experts device to device: both
    // want peer access, which every pair opts into once per process
    // argus (gnat): peer access stays OFF unless BASALT_PEER_ACCESS=1. On gnat the RTX 5070 Ti cannot peer at all and a
    // 5060 Ti pair's P2P copy has raised Xid 31; turning P2P on there is the owner's call, not a default. Off, every
    // copy goes the long way, which is how v0.3.2 ran.
    const char* pa = std::getenv("BASALT_PEER_ACCESS");
    const bool peer_access = pa && std::string(pa) == "1";
    if (gc.cards.size() > 1 && !peer_access)
        basalt::log::info("basalt", "peer access: off (BASALT_PEER_ACCESS=1 enables it between the cards a run uses)");
    if (gc.cards.size() > 1 && peer_access) {
        int tried = 0;
        const int pairs = basalt::core::enable_peer_access((int) gc.cards.size(), tried);
        if (tried > 0) basalt::log::info("basalt", "peer access: %d of %d ordered pairs enabled", pairs, tried);
    }
    return std::nullopt;
}

}  // namespace basalt::program::cli
