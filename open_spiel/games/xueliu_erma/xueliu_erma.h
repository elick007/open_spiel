// Copyright 2024 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_XUELIU_ERMA_H_
#define OPEN_SPIEL_GAMES_XUELIU_ERMA_H_

// Xueliu Erma (血流二麻) — a 2-player Sichuan-style mahjong variant.
//
// Rules reference: https://zh.moegirl.org.cn/二人麻将
//
// Two suits only (Wan 万 and Tiao 条), 9 ranks each, 4 copies = 72 tiles.
// Each player gets 13 tiles. Players draw and discard in turns. Opponent can
// Pong (碰), Kong (杠), or Hu (和) on a discard. No Chi (吃) in 2-player.
// Game ends on first Hu or wall exhaustion (draw).
//
// Scoring (fan):
//   Base Hu: 2, Dui Dui Hu: 4, Qing Yi Se: 6, Qi Dui: 6,
//   Qing Dui: 10, Jin Gou Diao: 10, Qing Jin Gou Diao: 16,
//   Qing Qi Dui: 16, Tian Hu: 64 (standalone).
// Roots: each 4-of-a-kind in hand = 1 root, doubles score.
//   Zi Mo: +1 root. Qiang Gang: +1 root. Gang Shang Kai Hua: +2 roots.
// Kong: Ming Gang 2 fan, An Gang 4 fan.

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace xueliu_erma {

// Game constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumSuits = 2;    // Wan (万), Tiao (条)
inline constexpr int kNumRanks = 9;    // 1..9
inline constexpr int kTilesPerKind = 4;
inline constexpr int kNumTileTypes = kNumSuits * kNumRanks;  // 18
inline constexpr int kNumTiles = kNumTileTypes * kTilesPerKind;  // 72
inline constexpr int kHandSize = 13;   // initial hand size
inline constexpr int kWallSize = kNumTiles - kNumPlayers * kHandSize;  // 46

// Tile encoding: tile_id = suit * kNumRanks + rank  (0..17)
// suit 0 = Wan, suit 1 = Tiao. rank 0..8 = 1..9.
inline int TileType(int suit, int rank) { return suit * kNumRanks + rank; }
inline int TileSuit(int tile_type) { return tile_type / kNumRanks; }
inline int TileRank(int tile_type) { return tile_type % kNumRanks; }

// For chance nodes: dealing action = position in shuffled deck (0..71).
// For player actions, we define the following action space:
//   Discard tile_type: 0..17
//   Pong:              18
//   Kong (declared):   19..36 (19 + tile_type for ming gang or an gang)
//   Hu:                37
//   Pass:              38
inline constexpr int kDiscardActionBase = 0;
inline constexpr int kPongAction = kNumTileTypes;           // 18
inline constexpr int kKongActionBase = kPongAction + 1;     // 19
inline constexpr int kHuAction = kKongActionBase + kNumTileTypes;  // 37
inline constexpr int kPassAction = kHuAction + 1;           // 38
inline constexpr int kNumDistinctActions = kPassAction + 1; // 39
inline constexpr int kNumNodeTypes = 6;
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

// Game phases.
enum class Phase {
  kDeal,      // Chance nodes dealing tiles
  kPlay,      // Normal play (draw/discard/response)
  kGameOver,  // Terminal
};

// Sub-phases within kPlay
enum class PlayPhase {
  kDraw,          // Current player draws a tile (chance node)
  kAfterDraw,     // Current player decides: discard, an-gang, or self-hu
  kAfterDiscard,  // Opponent decides: pong, hu, or pass
  kAfterAddKong,  // Opponent can rob an added kong, or allow it.
  kAfterKong,     // After a kong, draw replacement tile (chance node)
};

// Meld types for exposed melds.
enum class MeldType {
  kPong,     // 碰: 3 of a kind (exposed)
  kMingGang, // 明杠: exposed kong (from pong upgrade or direct)
  kAnGang,   // 暗杠: concealed kong (4 from hand)
};

struct Meld {
  MeldType type;
  int tile_type;
};

class XueliuErmaGame;

class XueliuErmaState : public State {
 public:
  XueliuErmaState(std::shared_ptr<const Game> game);
  XueliuErmaState(const XueliuErmaState&) = default;

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

  // Tile representation: hand_[player][tile_type] = count (0..4)
  std::array<std::array<int, kNumTileTypes>, kNumPlayers> hand_{};
  // Exposed melds for each player.
  std::array<std::vector<Meld>, kNumPlayers> melds_{};
  // Wall: remaining tiles to draw.
  std::vector<int> wall_;  // tile types, shuffled
  int wall_pos_ = 0;       // next position to draw from

  Phase phase_ = Phase::kDeal;
  PlayPhase play_phase_ = PlayPhase::kDraw;
  Player current_player_ = kChancePlayerId;
  int tiles_dealt_ = 0;
  int last_discard_ = -1;       // tile_type of last discard
  Player last_discard_player_ = kInvalidPlayer;
  bool is_gang_draw_ = false;   // true if drawing replacement after kong
  int last_drawn_tile_ = -1;    // tile_type of last drawn tile
  bool last_discard_after_kong_ = false;

  bool pending_add_kong_ = false;
  Player pending_kong_player_ = kInvalidPlayer;
  int pending_kong_tile_ = -1;
  std::array<std::vector<int>, kNumPlayers> discard_history_{};

  std::array<int, kNumPlayers> discard_count_{};
  int num_discards_total_ = 0;
  Action last_action_ = kInvalidAction;

  std::array<double, kNumPlayers> score_adjustments_{};

  // Scoring
  std::vector<double> returns_ = std::vector<double>(kNumPlayers, 0.0);
  int winner_ = kInvalidPlayer;
  bool is_self_draw_ = false;

  // Kong scoring accumulated per player (fan from ming/an gang)
  std::array<int, kNumPlayers> kong_fan_{};

  // Helper methods
  std::string TileTypeToString(int tile_type) const;
  bool CanHu(int player) const;
  bool CanHuWithTile(int player, int tile_type) const;
  bool CanHuWithAnyTile(int player) const;
  bool IsWinningHand(const std::array<int, kNumTileTypes>& h) const;
  bool IsSevenPairs(const std::array<int, kNumTileTypes>& h) const;
  bool CheckSetsAndPair(const std::array<int, kNumTileTypes>& h) const;
  bool IsDuiDuiHu(int player) const;
  bool IsMenQing(int player) const;
  bool IsZhongZhang(int player) const;
  bool IsJinGouHu(int player) const;
  bool IsQuanYaoJiu(int player) const;
  bool IsJiangDui(int player) const;
  bool IsJiangQiDui(int player) const;
  int CountConcealedTiles(int player) const;
  int CountPhysicalTiles(int player, int tile_type) const;
  bool HasWallTiles() const;
  int CurrentNodeType() const;
  void WriteObservationFeatures(Player player, absl::Span<float> values) const;
  void RecordPublicActionEvent(Player player, Action action);
  int CountRoots(int player) const;
  int BaseFan(int player) const;
  int HuFan(int player, const WinContext& context) const;
  int BonusFan(int player, const WinContext& context) const;
  void ScoreUp(int winner, const WinContext& context);
  void ClearPendingAddKong();
  void ClearDiscardContext();
  void SetDrawOutcome();

  // Legal action helpers
  std::vector<Action> DealLegalActions() const;
  std::vector<Action> DrawLegalActions() const;
  std::vector<Action> AfterDrawLegalActions() const;
  std::vector<Action> AfterDiscardLegalActions() const;
  std::vector<Action> AfterAddKongLegalActions() const;

  // Apply action helpers
  void ApplyDealAction(Action action);
  void ApplyDrawAction(Action action);
  void ApplyAfterDrawAction(Action action);
  void ApplyAfterDiscardAction(Action action);
  void ApplyAfterAddKongAction(Action action);
};

class XueliuErmaGame : public Game {
 public:
  explicit XueliuErmaGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }
  int MaxChanceOutcomes() const override { return kNumTiles; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<XueliuErmaState>(shared_from_this());
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -256; }
  double MaxUtility() const override { return 256; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    // Deal (72) + at most ~46 draw/discard cycles * ~3 actions each
    return kNumTiles + kWallSize * 3;
  }
};

}  // namespace xueliu_erma
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_XUELIU_ERMA_H_
