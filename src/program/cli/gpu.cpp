#include "gpu.hpp"

#include "basalt/log.hpp"
#include "basalt/program/options.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

namespace basalt::program::cli {

std::optional<int> select_gpus(CommonOptions& o, std::vector<GpuCard>& chosen_cards, int max_expert_tiers) {
    namespace bp = basalt::program;
    bp::GpuSpec spec;
    std::string gerr;
    if (!bp::parse_gpu_spec(o.gpu, spec, gerr)) {
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
    bp::GpuChoice choice = bp::choose_gpus(spec, visible, known);
    bp::choose_vision_card(choice, o.vision_gpu, visible.size(), known);
    if (o.gpu_plan) {
        std::string lossy;
        for (const auto& f : o.lossy) lossy += (lossy.empty() ? "" : ",") + f.flag;
        std::printf("%s\n", bp::gpu_plan_json(visible, choice, o.arch, o.quality_flags, o.quality_window, o.plan_spec, o.max_context,
                                             o.model_name, lossy).c_str());
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
    bp::RoleInputs ri;
    for (int i : choice.cards) ri.cards.push_back((size_t) i < visible.size() ? visible[(size_t) i] : bp::GpuCard{});
    ri.max_expert_tiers = max_expert_tiers;   // the device-planned tail's peer windows
    const std::vector<bp::GpuRole> roles = bp::choose_roles(ri);
    std::string line;
    for (size_t d = 0; d < roles.size(); ++d) {
        const bp::GpuCard& c = ri.cards[d];
        line += std::string(d ? ", " : "") + (c.name.empty() ? "GPU" : c.name) +
                (c.smi >= 0 ? " (nvidia-smi " + std::to_string(c.smi) + ")" : "") + " " + bp::gpu_role_name(roles[d]);
        if (roles[d] == bp::GpuRole::ExpertTier) {
            o.peer_devices.push_back((int) d);
            o.peer_gbps.push_back(c.mem_gbps);
        }
    }
    basalt::log::info("basalt", "gpus: %s", line.c_str());
    if (std::count(roles.begin(), roles.end(), bp::GpuRole::Idle) > 0)
        basalt::log::warn("basalt", "%s",
                          ("expert tiers max " + std::to_string(ri.max_expert_tiers) + "; later cards idle").c_str());
    if (!o.peer_devices.empty()) o.peer_device = o.peer_devices[0];
    chosen_cards = ri.cards;
    // The split hands a window's activations from one card to the next, and a tier card reads its experts d2d: both
    // want peer access, which every pair must opt into once per process.  A pair the driver refuses is left alone and
    // the copy goes the long way.
    // argus (gnat): peer access stays OFF unless BASALT_PEER_ACCESS=1. On gnat the RTX 5070 Ti cannot peer at all and a
    // 5060 Ti pair's P2P copy has raised Xid 31; turning P2P on there is the owner's call, not a default. Off, every
    // copy goes the long way, which is how v0.3.2 ran.
    const char* pa = std::getenv("BASALT_PEER_ACCESS");
    const bool peer_access = pa && std::string(pa) == "1";
    if (chosen_cards.size() > 1 && !peer_access)
        basalt::log::info("basalt", "peer access: off (BASALT_PEER_ACCESS=1 enables it between the cards a run uses)");
    if (chosen_cards.size() > 1 && peer_access) {
        int pairs = 0, tried = 0, prev = 0;
        cudaGetDevice(&prev);
        for (int a = 0; a < (int) chosen_cards.size(); ++a)
            for (int b = 0; b < (int) chosen_cards.size(); ++b) {
                int can = 0;
                if (a == b || cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) {
                    (void) cudaGetLastError();
                    continue;
                }
                ++tried;
                if (cudaSetDevice(a) != cudaSuccess) { (void) cudaGetLastError(); continue; }
                const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
                if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) ++pairs;
                (void) cudaGetLastError();
            }
        (void) cudaSetDevice(prev);
        if (tried > 0) basalt::log::info("basalt", "peer access: %d of %d ordered pairs enabled", pairs, tried);
    }
    return std::nullopt;
}

}  // namespace basalt::program::cli
