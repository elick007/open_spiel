// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_EREN_YIFANG_H_
#define OPEN_SPIEL_GAMES_EREN_YIFANG_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace eren_yifang {

inline constexpr int kNumPlayers = 2;
inline constexpr int kNumRanks = 9;
inline constexpr int kTilesPerKind = 4;
inline constexpr int kNumTileTypes = kNumRanks;
inline constexpr int kNumTiles = kNumTileTypes * kTilesPerKind;
inline constexpr int kWinningSetCount = 2;
inline constexpr int kDealerInitialHandSize = 8;
inline constexpr int kNonDealerInitialHandSize = 7;
inline constexpr int kInitialDealCount =
    kDealerInitialHandSize + kNonDealerInitialHandSize;
inline constexpr int kWallSize = kNumTiles - kInitialDealCount;

inline int TileRank(int tile_type) { return tile_type; }

inline constexpr int kDiscardActionBase = 0;
inline constexpr int kPongAction = kNumTileTypes;
inline constexpr int kKongActionBase = kPongAction + 1;
inline constexpr int kHuAction = kKongActionBase + kNumTileTypes;
inline constexpr int kPassAction = kHuAction + 1;
inline constexpr int kNumDistinctActions = kPassAction + 1;
inline constexpr int kNumNodeTypes = 7;
inline constexpr int kTensorMapHeight = 4;
inline constexpr int kTensorMapWidth = kNumTileTypes;
inline constexpr int kTileMapSize = kTensorMapHeight * kTensorMapWidth;
inline constexpr int kMaxTrackedDiscards = kWallSize;
inline constexpr int kImageLikeChannels = 5 + 2 * kMaxTrackedDiscards;
inline constexpr int kImageLikeFeatureSize = kTileMapSize * kImageLikeChannels;
inline constexpr int kTileOneHotWithNoneSize = kNumTileTypes + 1;
inline constexpr int kPositionFeatureSize = kNumPlayers;
inline constexpr int kSelfReadyFeatureSize = 2;
inline constexpr int kLastActionFeatureSize = kNumDistinctActions;
inline constexpr int kPublicFlagFeatureSize = 3;
inline constexpr int kScoreAdjustmentFeatureSize = kNumPlayers;
inline constexpr int kWallRemainingFeatureSize = 1;
inline constexpr int kSideFeatureSize =
    kPositionFeatureSize + kSelfReadyFeatureSize + kNumNodeTypes +
    kLastActionFeatureSize + 2 * kTileOneHotWithNoneSize +
    kPublicFlagFeatureSize + kScoreAdjustmentFeatureSize +
    kWallRemainingFeatureSize;
inline constexpr int kObservationTensorSize =
    kImageLikeFeatureSize + kSideFeatureSize;
inline constexpr int kInformationStateTensorSize = kObservationTensorSize;

enum class Phase {
  kDeal,
  kPlay,
  kGameOver,
};

enum class PlayPhase {
  kDraw,
  kAfterDraw,
  kAfterPong,
  kAfterDiscard,
  kAfterAddKong,
};

enum class MeldType {
  kPong,
  kMingGang,
  kAnGang,
};

struct Meld {
  MeldType type;
  int tile_type;
};

class ErenYifangGame;

class ErenYifangState : public State {
 public:
  explicit ErenYifangState(std::shared_ptr<const Game> game);
  ErenYifangState(const ErenYifangState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  struct WinContext {
    bool self_draw = false;
    bool gang_shang_kai_hua = false;
    bool gang_shang_pao = false;
    bool qiang_gang_hu = false;
    bool saodi_hu = false;
    bool jin_gou_hu = false;
    bool haidi_pao = false;
    bool tian_hu = false;
    bool di_hu = false;
    int winning_tile = -1;
  };

  std::array<std::array<int, kNumTileTypes>, kNumPlayers> hand_{};
  std::array<std::vector<Meld>, kNumPlayers> melds_{};
  std::array<std::vector<int>, kNumPlayers> discard_history_{};
  std::vector<int> wall_;
  int wall_pos_ = 0;

  Phase phase_ = Phase::kDeal;
  PlayPhase play_phase_ = PlayPhase::kAfterDraw;
  Player current_player_ = kChancePlayerId;
  int tiles_dealt_ = 0;
  int last_drawn_tile_ = -1;

  int last_discard_ = -1;
  Player last_discard_player_ = kInvalidPlayer;
  bool last_discard_after_kong_ = false;

  bool pending_add_kong_ = false;
  Player pending_kong_player_ = kInvalidPlayer;
  int pending_kong_tile_ = -1;

  bool rollbackable_kong_active_ = false;
  Player rollbackable_kong_player_ = kInvalidPlayer;
  int rollbackable_kong_points_ = 0;
  bool last_draw_was_kong_replacement_ = false;

  std::array<int, kNumPlayers> discard_count_{};
  Action last_action_ = kInvalidAction;
  int num_discards_total_ = 0;

  std::array<double, kNumPlayers> score_adjustments_{};
  std::vector<double> returns_ = std::vector<double>(kNumPlayers, 0.0);

  std::string TileTypeToString(int tile_type) const;
  int CountConcealedTiles(int player) const;
  int CountPhysicalTiles(int player, int tile_type) const;
  int RequiredConcealedSetCount(int player) const;
  bool HasWallTiles() const;
  int CurrentNodeType() const;
  void WriteObservationFeatures(Player player, absl::Span<float> values) const;
  void RecordPublicActionEvent(Player player, Action action);

  bool CanHu(int player) const;
  bool CanHuWithTile(int player, int tile_type) const;
  bool CanHuWithAnyTile(int player) const;
  bool IsWinningHand(const std::array<int, kNumTileTypes>& hand,
                     int meld_count) const;
  bool IsAllTriplets(int player) const;
  bool IsMenQing(int player) const;
  bool IsZhongZhang(int player) const;
  bool IsJinGouHu(int player) const;
  bool IsJiaXinFive(int player, int winning_tile) const;
  int CountRoots(int player) const;
  int BaseFan(int player) const;
  int BonusFan(int player, const WinContext& context) const;
  void ApplyScoreDelta(int player, int points);
  void SetRollbackableKong(int player, int points);
  void ClearRollbackableKong();
  void UndoRollbackableKong();
  void ClearPendingAddKong();
  void ClearDiscardContext();
  void SetDrawOutcome();
  void ScoreUp(int winner, const WinContext& context);

  std::vector<Action> DealLegalActions() const;
  std::vector<Action> DrawLegalActions() const;
  std::vector<Action> AfterDrawLegalActions() const;
  std::vector<Action> AfterPongLegalActions() const;
  std::vector<Action> AfterDiscardLegalActions() const;
  std::vector<Action> AfterAddKongLegalActions() const;

  void ApplyDealAction(Action action);
  void ApplyDrawAction(Action action);
  void ApplyAfterDrawAction(Action action);
  void ApplyAfterPongAction(Action action);
  void ApplyAfterDiscardAction(Action action);
  void ApplyAfterAddKongAction(Action action);
};

class ErenYifangGame : public Game {
 public:
  explicit ErenYifangGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  int MaxChanceOutcomes() const override { return kNumTiles; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<ErenYifangState>(shared_from_this());
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -512; }
  double MaxUtility() const override { return 512; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return 128; }
};

}  // namespace eren_yifang
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_EREN_YIFANG_H_
