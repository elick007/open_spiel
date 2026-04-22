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

inline constexpr int kDrawAction = 0;
inline constexpr int kHuAction = 1;
inline constexpr int kPassAction = 2;
inline constexpr int kDiscardActionBase = 3;
inline constexpr int kDiscardActionEnd = kDiscardActionBase + kNumTileTypes - 1;
inline constexpr int kPongActionBase = kDiscardActionEnd + 1;
inline constexpr int kPongActionEnd = kPongActionBase + kNumTileTypes - 1;
inline constexpr int kGongActionBase = kPongActionEnd + 1;
inline constexpr int kGongActionEnd = kGongActionBase + kNumTileTypes - 1;
inline constexpr int kConcealedGongActionBase = kGongActionEnd + 1;
inline constexpr int kConcealedGongActionEnd =
    kConcealedGongActionBase + kNumTileTypes - 1;
inline constexpr int kPassHuAction = kConcealedGongActionEnd + 1;
inline constexpr int kAddGongActionBase = kPassHuAction + 1;
inline constexpr int kAddGongActionEnd =
    kAddGongActionBase + kNumTileTypes - 1;
inline constexpr int kNumDistinctActions = kAddGongActionEnd + 1;

inline constexpr int kImageObservationChannels = 31;
inline constexpr int kPositionFeatureChannels = kNumPlayers;
inline constexpr int kLastActionFeatureChannels = kNumPlayers * kNumDistinctActions;
inline constexpr int kAuxObservationChannels =
    kPositionFeatureChannels + kLastActionFeatureChannels;
inline constexpr int kObservationChannels = kImageObservationChannels;
inline constexpr int kObservationHeight = 4;
inline constexpr int kObservationWidth = kNumTileTypes;
inline constexpr int kMaxTrackedDiscards = 13;
inline constexpr int kObservationTensorSize =
    kObservationChannels * kObservationHeight * kObservationWidth;
inline constexpr int kInformationStateTensorSize = kAuxObservationChannels;

enum class Phase {
  kDeal,
  kPlay,
  kGameOver,
};

enum class PlayPhase {
  kActorTurn,
  kRespondToDiscard,
  kRespondToAddGong,
  kAwaitDraw,
  kDrawChance,
};

enum class MeldType {
  kPong,
  kDirectGong,
  kAddGong,
  kConcealedGong,
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
  std::array<std::array<int, kNumTileTypes>, kNumPlayers> gong_mode_{};
  std::vector<int> wall_;
  int wall_pos_ = 0;

  Phase phase_ = Phase::kDeal;
  PlayPhase play_phase_ = PlayPhase::kActorTurn;
  Player current_player_ = kChancePlayerId;
  int tiles_dealt_ = 0;

  bool discard_only_turn_ = false;
  bool hu_declined_in_context_ = false;
  int last_drawn_tile_ = -1;

  int last_discard_ = -1;
  Player last_discard_player_ = kInvalidPlayer;

  bool pending_add_gong_ = false;
  Player pending_kong_player_ = kInvalidPlayer;
  int pending_kong_tile_ = -1;
  Player pending_draw_player_ = kInvalidPlayer;

  std::array<bool, kNumPlayers> is_first_action_{true, true};
  std::array<bool, kNumPlayers> is_gonging_{};
  std::array<bool, kNumPlayers> discard_after_gong_{};

  Action last_action_ = kInvalidAction;
  std::array<Action, kNumPlayers> last_actions_by_player_{
      {kInvalidAction, kInvalidAction}};
  std::vector<double> returns_ = std::vector<double>(kNumPlayers, 0.0);

  std::string TileTypeToString(int tile_type) const;
  int CountConcealedTiles(int player) const;
  int CountPhysicalTiles(int player, int tile_type) const;
  bool HasWallTiles() const;

  bool CanHu(int player) const;
  bool CanHuWithTile(int player, int tile_type) const;
  bool IsWinningHand(const std::array<int, kNumTileTypes>& hand,
                     int meld_count) const;
  bool IsDuiDuiHu(int player) const;
  bool IsMenQing(int player) const;
  bool IsZhongZhang(int player) const;
  bool IsJinGouHu(int player) const;
  bool IsJiaXinFive(int player, int winning_tile) const;
  int CountRoots(int player) const;
  int BaseFan(int player) const;
  int BonusFan(int player, const WinContext& context) const;
  int KongScore(int player) const;

  void WriteObservationFeatures(Player player, absl::Span<float> values) const;
  void WriteInformationStateFeatures(Player player,
                                     absl::Span<float> values) const;
  void RecordPublicActionEvent(Action action);
  void ClearDiscardContext();
  void ClearPendingAddGong();
  void UndoPendingAddGong();
  void EnterDrawChance(Player player);
  void SetDrawOutcome();
  void ScoreUp(int winner, const WinContext& context);

  std::vector<Action> DealLegalActions() const;
  std::vector<Action> ActorTurnLegalActions() const;
  std::vector<Action> RespondToDiscardLegalActions() const;
  std::vector<Action> RespondToAddGongLegalActions() const;
  std::vector<Action> AwaitDrawLegalActions() const;

  void ApplyDealAction(Action action);
  void ApplyDrawChanceAction(Action action);
  void ApplyActorTurnAction(Action action);
  void ApplyRespondToDiscardAction(Action action);
  void ApplyRespondToAddGongAction(Action action);
  void ApplyAwaitDrawAction(Action action);
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
  double MinUtility() const override { return -128; }
  double MaxUtility() const override { return 128; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return 128; }
};

}  // namespace eren_yifang
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_EREN_YIFANG_H_
