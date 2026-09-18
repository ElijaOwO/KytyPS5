#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEWAVEANALYSIS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEWAVEANALYSIS_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct WaterfallReadLaneProof {
	Value lane_condition;
	Value low_mask;
	Value high_mask;
	Value low_backedge;
	Value high_backedge;
	Block* header = nullptr;
	Block* initial_block = nullptr;
	Block* backedge_block = nullptr;
	Block* read_block = nullptr;

	// The recognized selector has an all-zero-mask fallback. The lane condition
	// is therefore guaranteed only when either current mask contains a bit.
	bool may_use_empty_mask_fallback = true;
};

// Recognizes the 64-lane AMDGPU waterfall selector formed from two ballot-mask
// halves. Every accepted loop-carried mask must be a bit-clearing subset of
// the original Ballot(condition), so a selected set bit identifies a lane for
// which lane_condition was true.
//
// This does not claim anything about the selector's fallback when both masks
// are zero. Callers must either prove that case unreachable or map unknown
// keys to a semantically safe default.
std::optional<WaterfallReadLaneProof> AnalyzeWaterfallReadLane(
    const Program& program, Value value);

// Proves that a specific use is reached only on executions where the waterfall
// mask feeding ReadLane is non-empty. This closes the selector's all-zero-mask
// fallback without evaluating ReadLane on the host.
bool ProveWaterfallReadLaneUseGuard(const Program& program,
                                    const WaterfallReadLaneProof& proof,
                                    const Inst& use);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEWAVEANALYSIS_H_ */
