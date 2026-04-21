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

#include "open_spiel/games/xueliu_erma/xueliu_erma.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace xueliu_erma {
namespace {

const char* kSuitNames[] = {"W", "T"};  // Wan, Tiao
const char* kSuitChinese[] = {"万", "条"};

const GameType kGameType{
    /*short_name=*/"xueliu_erma",
    /*long_name=*/"Xueliu Erma",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new XueliuErmaGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

void WriteTileCountMap(const std::array<int, kNumTileTypes>& counts,
                       absl::Span<float> values) {
  SPIEL_CHECK_EQ(values.size(), kTileMapSize);
  std::fill(values.begin(), values.end(), 0.0f);
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    const int copies = std::min(counts[tile], kTensorMapHeight);
    for (int copy = 0; copy < copies; ++copy) {
      values[copy * kTensorMapWidth + tile] = 1.0f;
    }
  }
}

void WriteSingleTileMap(int tile_type, absl::Span<float> values) {
  SPIEL_CHECK_EQ(values.size(), kTileMapSize);
  std::fill(values.begin(), values.end(), 0.0f);
  if (tile_type >= 0 && tile_type < kNumTileTypes) {
    values[tile_type] = 1.0f;
  }
}

void WriteCountIndicatorMap(int count, absl::Span<float> values) {
  SPIEL_CHECK_EQ(values.size(), kTileMapSize);
  std::fill(values.begin(), values.end(), 0.0f);
  const int clamped = std::min(std::max(count, 0), kTensorMapHeight);
  for (int row = 0; row < clamped; ++row) {
    for (int tile = 0; tile < kTensorMapWidth; ++tile) {
      values[row * kTensorMapWidth + tile] = 1.0f;
    }
  }
}

void WriteOrderedTileSequence(const std::vector<int>& tiles,
                              absl::Span<float> values) {
  SPIEL_CHECK_EQ(values.size(), kMaxTrackedDiscards * kTileMapSize);
  std::fill(values.begin(), values.end(), 0.0f);
  const int kept = std::min(static_cast<int>(tiles.size()), kMaxTrackedDiscards);
  const int start = static_cast<int>(tiles.size()) - kept;
  for (int index = 0; index < kept; ++index) {
    WriteSingleTileMap(tiles[start + index],
                       values.subspan(index * kTileMapSize, kTileMapSize));
  }
}

}  // namespace

// ---------- Utility functions ----------

std::string XueliuErmaState::TileTypeToString(int tile_type) const {
  int suit = TileSuit(tile_type);
  int rank = TileRank(tile_type);
  return absl::StrCat(rank + 1, kSuitNames[suit]);
}

// Check if a hand (count array) forms 7 pairs.
bool XueliuErmaState::IsSevenPairs(
    const std::array<int, kNumTileTypes>& h) const {
  int pairs = 0;
  for (int i = 0; i < kNumTileTypes; ++i) {
    if (h[i] % 2 != 0) return false;
    pairs += h[i] / 2;
  }
  return pairs == 7;
}

// Recursive check: can the hand form groups of 3 (sets/sequences)?
// This checks one suit at a time since we only have numbered tiles.
static bool CheckSuitGroups(std::array<int, kNumRanks>& suit_hand) {
  // Find first non-zero
  int first = -1;
  for (int r = 0; r < kNumRanks; ++r) {
    if (suit_hand[r] > 0) {
      first = r;
      break;
    }
  }
  if (first == -1) return true;  // All zero, success.

  // Try triplet (刻子)
  if (suit_hand[first] >= 3) {
    suit_hand[first] -= 3;
    if (CheckSuitGroups(suit_hand)) {
      suit_hand[first] += 3;
      return true;
    }
    suit_hand[first] += 3;
  }

  // Try sequence (顺子) — only if rank+2 exists
  if (first + 2 < kNumRanks && suit_hand[first] >= 1 &&
      suit_hand[first + 1] >= 1 && suit_hand[first + 2] >= 1) {
    suit_hand[first]--;
    suit_hand[first + 1]--;
    suit_hand[first + 2]--;
    if (CheckSuitGroups(suit_hand)) {
      suit_hand[first]++;
      suit_hand[first + 1]++;
      suit_hand[first + 2]++;
      return true;
    }
    suit_hand[first]++;
    suit_hand[first + 1]++;
    suit_hand[first + 2]++;
  }

  return false;
}

// Check if a hand forms 4 sets + 1 pair (standard win).
bool XueliuErmaState::CheckSetsAndPair(
    const std::array<int, kNumTileTypes>& h) const {
  // Try each tile type as the pair
  for (int pair_tile = 0; pair_tile < kNumTileTypes; ++pair_tile) {
    if (h[pair_tile] < 2) continue;

    // Create a mutable copy with pair removed
    std::array<int, kNumTileTypes> remaining = h;
    remaining[pair_tile] -= 2;

    // Check each suit independently
    bool valid = true;
    for (int suit = 0; suit < kNumSuits; ++suit) {
      std::array<int, kNumRanks> suit_hand{};
      for (int r = 0; r < kNumRanks; ++r) {
        suit_hand[r] = remaining[suit * kNumRanks + r];
      }
      if (!CheckSuitGroups(suit_hand)) {
        valid = false;
        break;
      }
    }
    if (valid) return true;
  }
  return false;
}

// Check if the hand of a player is a winning hand.
bool XueliuErmaState::IsWinningHand(
    const std::array<int, kNumTileTypes>& h) const {
  return IsSevenPairs(h) || CheckSetsAndPair(h);
}

// Can a player hu with their current hand?
bool XueliuErmaState::CanHu(int player) const {
  return IsWinningHand(hand_[player]);
}

// Can a player hu if they add tile_type to their hand?
bool XueliuErmaState::CanHuWithTile(int player, int tile_type) const {
  auto h = hand_[player];
  h[tile_type]++;
  return IsWinningHand(h);
}

bool XueliuErmaState::CanHuWithAnyTile(int player) const {
  for (int tile_type = 0; tile_type < kNumTileTypes; ++tile_type) {
    if (CanHuWithTile(player, tile_type)) return true;
  }
  return false;
}

bool XueliuErmaState::IsDuiDuiHu(int player) const {
  const auto& h = hand_[player];
  for (int pair_tile = 0; pair_tile < kNumTileTypes; ++pair_tile) {
    if (h[pair_tile] < 2) continue;
    auto remaining = h;
    remaining[pair_tile] -= 2;
    bool all_triplets = true;
    for (int i = 0; i < kNumTileTypes; ++i) {
      if (remaining[i] % 3 != 0) {
        all_triplets = false;
        break;
      }
    }
    if (all_triplets) return true;
  }
  return false;
}

bool XueliuErmaState::IsMenQing(int player) const {
  for (const auto& meld : melds_[player]) {
    if (meld.type != MeldType::kAnGang) return false;
  }
  return true;
}

bool XueliuErmaState::IsZhongZhang(int player) const {
  auto is_terminal_rank = [](int tile_type) {
    int rank = TileRank(tile_type);
    return rank == 0 || rank == kNumRanks - 1;
  };

  for (int tile_type = 0; tile_type < kNumTileTypes; ++tile_type) {
    if (hand_[player][tile_type] > 0 && is_terminal_rank(tile_type)) {
      return false;
    }
  }
  for (const auto& meld : melds_[player]) {
    if (is_terminal_rank(meld.tile_type)) {
      return false;
    }
  }
  return true;
}

bool XueliuErmaState::IsJinGouHu(int player) const {
  int hand_tile_count = 0;
  for (int i = 0; i < kNumTileTypes; ++i) hand_tile_count += hand_[player][i];
  return hand_tile_count == 2 && melds_[player].size() == 4;
}

bool XueliuErmaState::IsQuanYaoJiu(int player) const {
  auto is_terminal_rank = [](int tile_type) {
    int rank = TileRank(tile_type);
    return rank == 0 || rank == kNumRanks - 1;
  };

  for (int tile_type = 0; tile_type < kNumTileTypes; ++tile_type) {
    if (hand_[player][tile_type] > 0 && !is_terminal_rank(tile_type)) {
      return false;
    }
  }
  for (const auto& meld : melds_[player]) {
    if (!is_terminal_rank(meld.tile_type)) {
      return false;
    }
  }
  return true;
}

bool XueliuErmaState::IsJiangDui(int player) const {
  if (!IsDuiDuiHu(player)) return false;
  for (int tile_type = 0; tile_type < kNumTileTypes; ++tile_type) {
    if (hand_[player][tile_type] > 0) {
      int rank = TileRank(tile_type) + 1;
      if (rank != 2 && rank != 5 && rank != 8) return false;
    }
  }
  for (const auto& meld : melds_[player]) {
    int rank = TileRank(meld.tile_type) + 1;
    if (rank != 2 && rank != 5 && rank != 8) return false;
  }
  return true;
}

bool XueliuErmaState::IsJiangQiDui(int player) const {
  if (!IsSevenPairs(hand_[player])) return false;
  for (int tile_type = 0; tile_type < kNumTileTypes; ++tile_type) {
    if (hand_[player][tile_type] > 0) {
      int rank = TileRank(tile_type) + 1;
      if (rank != 2 && rank != 5 && rank != 8) return false;
    }
  }
  return true;
}

int XueliuErmaState::CountConcealedTiles(int player) const {
  return std::accumulate(hand_[player].begin(), hand_[player].end(), 0);
}

int XueliuErmaState::CountPhysicalTiles(int player, int tile_type) const {
  int count = hand_[player][tile_type];
  for (const auto& meld : melds_[player]) {
    if (meld.tile_type != tile_type) continue;
    count += (meld.type == MeldType::kPong) ? 3 : 4;
  }
  return count;
}

bool XueliuErmaState::HasWallTiles() const {
  return wall_pos_ < static_cast<int>(wall_.size());
}

int XueliuErmaState::CurrentNodeType() const {
  if (phase_ == Phase::kDeal) return 0;
  if (phase_ == Phase::kGameOver) return 5;
  switch (play_phase_) {
    case PlayPhase::kDraw:
    case PlayPhase::kAfterKong:
      return 1;
    case PlayPhase::kAfterDraw:
      return 2;
    case PlayPhase::kAfterDiscard:
      return 3;
    case PlayPhase::kAfterAddKong:
      return 4;
  }
  SpielFatalError("Unknown node type.");
}

void XueliuErmaState::WriteObservationFeatures(Player player,
                                              absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
  std::fill(values.begin(), values.end(), 0.0f);

  const int opponent = 1 - player;

  std::array<int, kNumTileTypes> self_exposed{};
  std::array<int, kNumTileTypes> self_concealed_kongs{};
  for (const auto& meld : melds_[player]) {
    if (meld.type == MeldType::kAnGang) {
      self_concealed_kongs[meld.tile_type] += kTilesPerKind;
    } else {
      self_exposed[meld.tile_type] +=
          (meld.type == MeldType::kPong) ? 3 : kTilesPerKind;
    }
  }

  std::array<int, kNumTileTypes> opponent_exposed{};
  int opponent_concealed_kong_count = 0;
  for (const auto& meld : melds_[opponent]) {
    if (meld.type == MeldType::kAnGang) {
      ++opponent_concealed_kong_count;
    } else {
      opponent_exposed[meld.tile_type] +=
          (meld.type == MeldType::kPong) ? 3 : kTilesPerKind;
    }
  }

  int offset = 0;
  auto write_map = [&](const std::array<int, kNumTileTypes>& counts) {
    WriteTileCountMap(counts, values.subspan(offset, kTileMapSize));
    offset += kTileMapSize;
  };

  write_map(hand_[player]);
  write_map(self_exposed);
  write_map(self_concealed_kongs);
  WriteOrderedTileSequence(
      discard_history_[player],
      values.subspan(offset, kMaxTrackedDiscards * kTileMapSize));
  offset += kMaxTrackedDiscards * kTileMapSize;

  write_map(opponent_exposed);
  WriteCountIndicatorMap(opponent_concealed_kong_count,
                         values.subspan(offset, kTileMapSize));
  offset += kTileMapSize;

  WriteOrderedTileSequence(
      discard_history_[opponent],
      values.subspan(offset, kMaxTrackedDiscards * kTileMapSize));
  offset += kMaxTrackedDiscards * kTileMapSize;
  SPIEL_CHECK_EQ(offset, kImageLikeFeatureSize);

  auto side = values.subspan(kImageLikeFeatureSize, kSideFeatureSize);
  int side_offset = 0;

  side[side_offset + player] = 1.0f;
  side_offset += kPositionFeatureSize;

  const int self_ready_index = CanHuWithAnyTile(player) ? 1 : 0;
  side[side_offset + self_ready_index] = 1.0f;
  side_offset += kSelfReadyFeatureSize;

  side[side_offset + CurrentNodeType()] = 1.0f;
  side_offset += kNumNodeTypes;

  if (last_action_ >= 0 && last_action_ < kNumDistinctActions) {
    side[side_offset + last_action_] = 1.0f;
  }
  side_offset += kLastActionFeatureSize;

  if (pending_kong_tile_ >= 0) {
    side[side_offset + pending_kong_tile_] = 1.0f;
  } else {
    side[side_offset + kNumTileTypes] = 1.0f;
  }
  side_offset += kTileOneHotWithNoneSize;

  const bool last_draw_is_visible =
      phase_ == Phase::kPlay && play_phase_ == PlayPhase::kAfterDraw &&
      current_player_ == player && last_drawn_tile_ >= 0;
  if (last_draw_is_visible) {
    side[side_offset + last_drawn_tile_] = 1.0f;
  } else {
    side[side_offset + kNumTileTypes] = 1.0f;
  }
  side_offset += kTileOneHotWithNoneSize;

  side[side_offset++] = pending_add_kong_ ? 1.0f : 0.0f;
  side[side_offset++] = last_discard_after_kong_ ? 1.0f : 0.0f;
  side[side_offset++] = is_gang_draw_ ? 1.0f : 0.0f;

  for (int p = 0; p < kNumPlayers; ++p) {
    side[side_offset++] = static_cast<float>(score_adjustments_[p]);
  }
  side[side_offset++] =
      static_cast<float>(wall_.size() - wall_pos_) / kWallSize;
  SPIEL_CHECK_EQ(side_offset, kSideFeatureSize);
}

void XueliuErmaState::RecordPublicActionEvent(Player player, Action action) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  last_action_ = action;
}

// Count roots (根): each set of 4 identical tiles in hand.
int XueliuErmaState::CountRoots(int player) const {
  int roots = 0;
  for (int i = 0; i < kNumTileTypes; ++i) {
    if (CountPhysicalTiles(player, i) == 4) roots++;
  }
  return roots;
}

int XueliuErmaState::BaseFan(int player) const {
  const auto& h = hand_[player];
  bool is_seven_pairs = IsSevenPairs(h);
  bool is_dui_dui = !is_seven_pairs && IsDuiDuiHu(player);
  bool is_long_qi_dui = is_seven_pairs && CountRoots(player) > 0;
  bool is_qing = true;
  int found_suit = -1;
  for (int i = 0; i < kNumTileTypes; ++i) {
    if (h[i] > 0) {
      int s = TileSuit(i);
      if (found_suit == -1) found_suit = s;
      else if (s != found_suit) { is_qing = false; break; }
    }
  }
  if (is_qing) {
    for (const auto& meld : melds_[player]) {
      int s = TileSuit(meld.tile_type);
      if (found_suit == -1) found_suit = s;
      else if (s != found_suit) { is_qing = false; break; }
    }
  }

  bool is_quan_yao_jiu = IsQuanYaoJiu(player);
  bool is_jiang_dui = is_dui_dui && IsJiangDui(player);
  bool is_jiang_qi_dui = is_seven_pairs && IsJiangQiDui(player);
  int fan = 0;
  if (IsMenQing(player))
  {
    ++fan;
  }
  if (IsZhongZhang(player))
  {
    ++fan;
  }

  if (is_jiang_dui)
  {
    fan += 3;
  }else if (is_dui_dui)
  {
    ++fan;
  }

  if (is_qing)
  {
    fan += 2;
  }

  if (is_jiang_qi_dui)
  {
    fan += 4;
  }else if (is_long_qi_dui)
  {
    fan += 3;
  }else if (is_seven_pairs)
  {
    fan += 2;
  }

  if (is_quan_yao_jiu)
  {
    fan += 3;
  }

  return fan;
}

int XueliuErmaState::BonusFan(int player, const WinContext& context) const {
  int bonus = CountRoots(player);
  if (context.self_draw) ++bonus;
  if (context.gang_shang_kai_hua) ++bonus;
  if (context.gang_shang_pao) ++bonus;
  if (context.qiang_gang_hu) ++bonus;
  if (context.saodi_hu) ++bonus;
  if (context.jin_gou_hu) ++bonus;
  if (context.haidi_pao) ++bonus;
  if (context.tian_hu) bonus += 3;
  if (context.di_hu) bonus += 2;
  return bonus;
}

int XueliuErmaState::HuFan(int player, const WinContext& context) const {
  return std::min(BaseFan(player) + BonusFan(player, context), 4);
}

void XueliuErmaState::ScoreUp(int winner, const WinContext& context) {
  winner_ = winner;
  is_self_draw_ = context.self_draw;
  const int loser = 1 - winner;
  const int total_fan = HuFan(winner, context);
  double hu_score = std::ldexp(1.0, total_fan);

  returns_[0] = score_adjustments_[0];
  returns_[1] = score_adjustments_[1];
  returns_[winner] += hu_score;
  returns_[loser] -= hu_score;
}

void XueliuErmaState::SetDrawOutcome() {
  bool player0_ready = CanHuWithAnyTile(0);
  bool player1_ready = CanHuWithAnyTile(1);

  if (player0_ready == player1_ready) {
    returns_[0] = score_adjustments_[0];
    returns_[1] = score_adjustments_[1];
    phase_ = Phase::kGameOver;
    return;
  }

  int winner = player0_ready ? 0 : 1;
  int loser = 1 - winner;
  returns_[0] = score_adjustments_[0];
  returns_[1] = score_adjustments_[1];
  double score = 16.0;
  returns_[winner] += score;
  returns_[loser] -= score;
  phase_ = Phase::kGameOver;
}

// ---------- XueliuErmaState ----------

XueliuErmaState::XueliuErmaState(std::shared_ptr<const Game> game)
    : State(game) {
  // Initialize wall with all 72 tiles
  wall_.reserve(kNumTiles);
  for (int tile_type = 0; tile_type < kNumTileTypes; ++tile_type) {
    for (int copy = 0; copy < kTilesPerKind; ++copy) {
      wall_.push_back(tile_type);
    }
  }
}

Player XueliuErmaState::CurrentPlayer() const {
  if (phase_ == Phase::kGameOver) return kTerminalPlayerId;
  if (phase_ == Phase::kDeal) return kChancePlayerId;
  if (play_phase_ == PlayPhase::kDraw || play_phase_ == PlayPhase::kAfterKong) {
    return kChancePlayerId;
  }
  return current_player_;
}

std::string XueliuErmaState::ActionToString(Player player,
                                             Action action) const {
  if (player == kChancePlayerId) {
    if (phase_ == Phase::kDeal) {
      int tile_type = wall_[action];
      return absl::StrCat("Deal ", TileTypeToString(tile_type));
    } else {
      // Draw from wall
      int tile_type = wall_[action];
      return absl::StrCat("Draw ", TileTypeToString(tile_type));
    }
  }
  if (action >= kDiscardActionBase && action < kPongAction) {
    return absl::StrCat("Discard ", TileTypeToString(action));
  }
  if (action == kPongAction) return "Pong";
  if (action >= kKongActionBase && action < kHuAction) {
    int tile_type = action - kKongActionBase;
    return absl::StrCat("Kong ", TileTypeToString(tile_type));
  }
  if (action == kHuAction) return "Hu";
  if (action == kPassAction) return "Pass";
  return absl::StrCat("Unknown(", action, ")");
}

std::string XueliuErmaState::ToString() const {
  std::string rv;
  for (int p = 0; p < kNumPlayers; ++p) {
    absl::StrAppend(&rv, "Player ", p, " hand: ");
    for (int i = 0; i < kNumTileTypes; ++i) {
      for (int c = 0; c < hand_[p][i]; ++c) {
        absl::StrAppend(&rv, TileTypeToString(i), " ");
      }
    }
    if (!melds_[p].empty()) {
      absl::StrAppend(&rv, " Melds: ");
      for (const auto& m : melds_[p]) {
        switch (m.type) {
          case MeldType::kPong:
            absl::StrAppend(&rv, "[Pong ", TileTypeToString(m.tile_type), "] ");
            break;
          case MeldType::kMingGang:
            absl::StrAppend(&rv, "[MingGang ", TileTypeToString(m.tile_type),
                            "] ");
            break;
          case MeldType::kAnGang:
            absl::StrAppend(&rv, "[AnGang ", TileTypeToString(m.tile_type),
                            "] ");
            break;
        }
      }
    }
    absl::StrAppend(&rv, "\n");
  }
  absl::StrAppend(&rv, "Wall remaining: ", wall_.size() - wall_pos_, "\n");
  if (IsTerminal()) {
    absl::StrAppend(&rv, "Result: P0=", returns_[0], " P1=", returns_[1],
                     "\n");
  }
  return rv;
}

bool XueliuErmaState::IsTerminal() const {
  return phase_ == Phase::kGameOver;
}

std::vector<double> XueliuErmaState::Returns() const {
  return returns_;
}

std::string XueliuErmaState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ObservationString(player);
}

std::string XueliuErmaState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  const int opponent = 1 - player;
  std::string rv = absl::StrFormat("P%d hand:", player);
  for (int i = 0; i < kNumTileTypes; ++i) {
    for (int c = 0; c < hand_[player][i]; ++c) {
      absl::StrAppend(&rv, " ", TileTypeToString(i));
    }
  }
  bool has_self_public_meld = false;
  int opponent_concealed_kongs = 0;
  for (const auto& meld : melds_[player]) {
    if (!has_self_public_meld) {
      absl::StrAppend(&rv, " | Self melds:");
      has_self_public_meld = true;
    }
    const char* label = nullptr;
    switch (meld.type) {
      case MeldType::kPong:
        label = "Pong";
        break;
      case MeldType::kMingGang:
        label = "MingGang";
        break;
      case MeldType::kAnGang:
        label = "AnGang";
        break;
    }
    absl::StrAppend(&rv, " [", label, " ", TileTypeToString(meld.tile_type),
                    "]");
  }
  if (!discard_history_[player].empty()) {
    absl::StrAppend(&rv, " | Self discards:");
    for (int tile_type : discard_history_[player]) {
      absl::StrAppend(&rv, " ", TileTypeToString(tile_type));
    }
  }
  bool has_opponent_exposed_meld = false;
  for (const auto& meld : melds_[opponent]) {
    if (meld.type == MeldType::kAnGang) {
      ++opponent_concealed_kongs;
      continue;
    }
    if (!has_opponent_exposed_meld) {
      absl::StrAppend(&rv, " | Opp exposed:");
      has_opponent_exposed_meld = true;
    }
    const char* label = meld.type == MeldType::kPong ? "Pong" : "MingGang";
    absl::StrAppend(&rv, " [", label, " ", TileTypeToString(meld.tile_type),
                    "]");
  }
  if (opponent_concealed_kongs > 0) {
    absl::StrAppend(&rv, " | Opp an-gang count:", opponent_concealed_kongs);
  }
  if (!discard_history_[opponent].empty()) {
    absl::StrAppend(&rv, " | Opp discards:");
    for (int tile_type : discard_history_[opponent]) {
      absl::StrAppend(&rv, " ", TileTypeToString(tile_type));
    }
  }
  absl::StrAppend(&rv, " | Wall: ", wall_.size() - wall_pos_);
  if (last_discard_ >= 0) {
    absl::StrAppend(&rv, " | Last discard: ", TileTypeToString(last_discard_),
                    " by P", last_discard_player_);
  }
  if (last_action_ >= 0 && last_action_ < kNumDistinctActions) {
    absl::StrAppend(&rv, " | Last action: ",
                    ActionToString(/*player=*/0, last_action_));
  }
  return rv;
}

void XueliuErmaState::InformationStateTensor(Player player,
                                             absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_EQ(values.size(), kInformationStateTensorSize);
  WriteObservationFeatures(player, values);
}

void XueliuErmaState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorSize());
  WriteObservationFeatures(player, values);
}

std::vector<std::pair<Action, double>> XueliuErmaState::ChanceOutcomes()
    const {
  std::vector<std::pair<Action, double>> outcomes;

  if (phase_ == Phase::kDeal) {
    // During dealing, each remaining position in the wall is equally likely.
    int remaining = kNumTiles - tiles_dealt_;
    double prob = 1.0 / remaining;
    for (int i = tiles_dealt_; i < kNumTiles; ++i) {
      outcomes.emplace_back(i, prob);
    }
  } else {
    // Drawing from wall: each remaining tile position equally likely.
    int remaining = static_cast<int>(wall_.size()) - wall_pos_;
    SPIEL_CHECK_GT(remaining, 0);
    double prob = 1.0 / remaining;
    for (int i = wall_pos_; i < static_cast<int>(wall_.size()); ++i) {
      outcomes.emplace_back(i, prob);
    }
  }
  return outcomes;
}

std::unique_ptr<State> XueliuErmaState::Clone() const {
  return std::make_unique<XueliuErmaState>(*this);
}

// ---------- Legal Actions ----------

std::vector<Action> XueliuErmaState::LegalActions() const {
  switch (phase_) {
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kPlay:
      switch (play_phase_) {
        case PlayPhase::kDraw:
        case PlayPhase::kAfterKong:
          return DrawLegalActions();
        case PlayPhase::kAfterDraw:
          return AfterDrawLegalActions();
        case PlayPhase::kAfterDiscard:
          return AfterDiscardLegalActions();
        case PlayPhase::kAfterAddKong:
          return AfterAddKongLegalActions();
      }
      break;
    case Phase::kGameOver:
      return {};
  }
  return {};
}

std::vector<Action> XueliuErmaState::DealLegalActions() const {
  std::vector<Action> actions;
  int remaining = kNumTiles - tiles_dealt_;
  actions.reserve(remaining);
  for (int i = tiles_dealt_; i < kNumTiles; ++i) {
    actions.push_back(i);
  }
  return actions;
}

std::vector<Action> XueliuErmaState::DrawLegalActions() const {
  std::vector<Action> actions;
  int remaining = static_cast<int>(wall_.size()) - wall_pos_;
  actions.reserve(remaining);
  for (int i = wall_pos_; i < static_cast<int>(wall_.size()); ++i) {
    actions.push_back(i);
  }
  return actions;
}

std::vector<Action> XueliuErmaState::AfterDrawLegalActions() const {
  std::vector<Action> actions;

  // Check self Hu (自摸)
  if (CanHu(current_player_)) {
    actions.push_back(kHuAction);
  }

  // Check An Gang (暗杠) — 4 of a kind in hand
  for (int i = 0; i < kNumTileTypes; ++i) {
    if (hand_[current_player_][i] == 4) {
      actions.push_back(kKongActionBase + i);
    }
  }

  // Check Jia Gang (加杠) — upgrade a pong to kong with tile from hand
  for (const auto& meld : melds_[current_player_]) {
    if (meld.type == MeldType::kPong &&
        hand_[current_player_][meld.tile_type] >= 1) {
      actions.push_back(kKongActionBase + meld.tile_type);
    }
  }

  // Discard a tile
  for (int i = 0; i < kNumTileTypes; ++i) {
    if (hand_[current_player_][i] > 0) {
      actions.push_back(kDiscardActionBase + i);
    }
  }

  // Sort and deduplicate
  std::sort(actions.begin(), actions.end());
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());

  return actions;
}

std::vector<Action> XueliuErmaState::AfterDiscardLegalActions() const {
  std::vector<Action> actions;
  int opponent = 1 - last_discard_player_;

  // Check Hu (和)
  if (CanHuWithTile(opponent, last_discard_)) {
    auto h = hand_[opponent];
    h[last_discard_]++;
    XueliuErmaState trial_state(*this);
    trial_state.hand_[opponent] = h;
    WinContext context;
    context.gang_shang_pao = last_discard_after_kong_;
    context.haidi_pao = (wall_pos_ == static_cast<int>(wall_.size()));
    context.jin_gou_hu = trial_state.IsJinGouHu(opponent);
    context.di_hu =
        (opponent == 1 && discard_count_[1] == 0 && discard_count_[0] == 1);
    context.winning_tile = last_discard_;
    if (trial_state.HuFan(opponent, context) >= 1) {
      actions.push_back(kHuAction);
    }
  }

  // Check Pong (碰)
  if (hand_[opponent][last_discard_] >= 2) {
    actions.push_back(kPongAction);
  }

  // Check Ming Gang (明杠) — 3 in hand + 1 discard
  if (hand_[opponent][last_discard_] >= 3) {
    actions.push_back(kKongActionBase + last_discard_);
  }

  // Pass — always an option
  actions.push_back(kPassAction);

  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> XueliuErmaState::AfterAddKongLegalActions() const {
  std::vector<Action> actions;
  if (CanHuWithTile(current_player_, pending_kong_tile_)) {
    actions.push_back(kHuAction);
  }
  actions.push_back(kPassAction);
  return actions;
}

// ---------- Apply Actions ----------

void XueliuErmaState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      ApplyDealAction(action);
      return;
    case Phase::kPlay:
      switch (play_phase_) {
        case PlayPhase::kDraw:
        case PlayPhase::kAfterKong:
          ApplyDrawAction(action);
          return;
        case PlayPhase::kAfterDraw:
          ApplyAfterDrawAction(action);
          return;
        case PlayPhase::kAfterDiscard:
          ApplyAfterDiscardAction(action);
          return;
        case PlayPhase::kAfterAddKong:
          ApplyAfterAddKongAction(action);
          return;
      }
      break;
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
  }
}

void XueliuErmaState::ApplyDealAction(Action action) {
  // action = position in the remaining undealt tiles. Swap with tiles_dealt_.
  SPIEL_CHECK_GE(action, tiles_dealt_);
  SPIEL_CHECK_LT(action, kNumTiles);

  // Swap wall[tiles_dealt_] and wall[action] to simulate Fisher-Yates shuffle
  std::swap(wall_[tiles_dealt_], wall_[action]);

  int tile_type = wall_[tiles_dealt_];
  int player = tiles_dealt_ % kNumPlayers;
  hand_[player][tile_type]++;
  tiles_dealt_++;

  if (tiles_dealt_ == kNumPlayers * kHandSize) {
    // Dealing complete. Player 0 draws first.
    phase_ = Phase::kPlay;
    play_phase_ = PlayPhase::kDraw;
    current_player_ = 0;
    wall_pos_ = tiles_dealt_;  // remaining wall starts here
  }
}

void XueliuErmaState::ApplyDrawAction(Action action) {
  // action = position in wall to draw from. Swap with wall_pos_.
  SPIEL_CHECK_GE(action, wall_pos_);
  SPIEL_CHECK_LT(action, static_cast<int>(wall_.size()));

  std::swap(wall_[wall_pos_], wall_[action]);
  int tile_type = wall_[wall_pos_];
  wall_pos_++;

  hand_[current_player_][tile_type]++;
  last_drawn_tile_ = tile_type;

  // Move to AfterDraw where the player decides what to do.
  play_phase_ = PlayPhase::kAfterDraw;
}

void XueliuErmaState::ApplyAfterDrawAction(Action action) {
  if (action == kHuAction) {
    RecordPublicActionEvent(current_player_, action);
    WinContext context;
    context.self_draw = true;
    context.gang_shang_kai_hua = is_gang_draw_;
    context.saodi_hu = (wall_pos_ == static_cast<int>(wall_.size()));
    context.jin_gou_hu = IsJinGouHu(current_player_);
    context.tian_hu = (current_player_ == 0 && num_discards_total_ == 0);
    context.di_hu =
        (current_player_ == 1 && discard_count_[1] == 0 && discard_count_[0] == 1);
    context.winning_tile = last_drawn_tile_;
    ScoreUp(current_player_, context);
    phase_ = Phase::kGameOver;
    return;
  }

  if (action >= kKongActionBase && action < kHuAction) {
    RecordPublicActionEvent(current_player_, action);
    // Kong action.
    int tile_type = action - kKongActionBase;

    // Check if it's an An Gang (暗杠) — 4 in hand
    if (hand_[current_player_][tile_type] == 4) {
      hand_[current_player_][tile_type] = 0;
      melds_[current_player_].push_back({MeldType::kAnGang, tile_type});
      kong_fan_[current_player_] += 2;
      ClearPendingAddKong();
      ClearDiscardContext();
      is_gang_draw_ = true;
      if (wall_pos_ >= static_cast<int>(wall_.size())) {
        SetDrawOutcome();
        return;
      }
      play_phase_ = PlayPhase::kAfterKong;
    } else {
      // Jia Gang (加杠), can be robbed.
      hand_[current_player_][tile_type]--;
      pending_add_kong_ = true;
      pending_kong_player_ = current_player_;
      pending_kong_tile_ = tile_type;
      current_player_ = 1 - current_player_;
      play_phase_ = PlayPhase::kAfterAddKong;
      return;
    }
    return;
  }

  // Discard action
  SPIEL_CHECK_GE(action, kDiscardActionBase);
  SPIEL_CHECK_LT(action, kPongAction);
  int tile_type = action - kDiscardActionBase;
  SPIEL_CHECK_GT(hand_[current_player_][tile_type], 0);

  RecordPublicActionEvent(current_player_, action);

  hand_[current_player_][tile_type]--;
  last_discard_ = tile_type;
  last_discard_player_ = current_player_;
  last_discard_after_kong_ = is_gang_draw_;
  is_gang_draw_ = false;
  discard_history_[current_player_].push_back(tile_type);
  ++discard_count_[current_player_];
  ++num_discards_total_;

  // Opponent gets to respond
  play_phase_ = PlayPhase::kAfterDiscard;
  current_player_ = 1 - current_player_;
}

void XueliuErmaState::ApplyAfterDiscardAction(Action action) {
  int opponent = current_player_;  // The one who is responding

  if (action == kHuAction) {
    RecordPublicActionEvent(opponent, action);
    // Win on opponent's discard
    hand_[opponent][last_discard_]++;
    WinContext context;
    context.gang_shang_pao = last_discard_after_kong_;
    context.haidi_pao = (wall_pos_ == static_cast<int>(wall_.size()));
    context.jin_gou_hu = IsJinGouHu(opponent);
    context.di_hu =
        (opponent == 1 && discard_count_[1] == 0 && discard_count_[0] == 1);
    context.winning_tile = last_discard_;
    ScoreUp(opponent, context);
    phase_ = Phase::kGameOver;
    return;
  }

  if (action == kPongAction) {
    RecordPublicActionEvent(opponent, action);
    // Pong: take 2 from hand + 1 from discard
    SPIEL_CHECK_GE(hand_[opponent][last_discard_], 2);
    hand_[opponent][last_discard_] -= 2;
    melds_[opponent].push_back({MeldType::kPong, last_discard_});
    ClearDiscardContext();

    // After pong, opponent becomes current player and must discard.
    current_player_ = opponent;
    play_phase_ = PlayPhase::kAfterDraw;  // Needs to discard (like after draw)
    return;
  }

  if (action >= kKongActionBase && action < kHuAction) {
    RecordPublicActionEvent(opponent, action);
    // Ming Gang: take 3 from hand + 1 from discard
    int tile_type = action - kKongActionBase;
    SPIEL_CHECK_EQ(tile_type, last_discard_);
    SPIEL_CHECK_GE(hand_[opponent][tile_type], 3);
    hand_[opponent][tile_type] -= 3;
    melds_[opponent].push_back({MeldType::kMingGang, tile_type});
    kong_fan_[opponent] += 2;
    ClearDiscardContext();

    // After kong, draw replacement.
    current_player_ = opponent;
    is_gang_draw_ = true;
    if (wall_pos_ >= static_cast<int>(wall_.size())) {
      SetDrawOutcome();
      return;
    }
    play_phase_ = PlayPhase::kDraw;
    return;
  }

  if (action == kPassAction) {
    RecordPublicActionEvent(opponent, action);
    // Pass: opponent doesn't react. Current player draws.
    current_player_ = opponent;  // The responder now draws
    ClearDiscardContext();

    if (wall_pos_ >= static_cast<int>(wall_.size())) {
      SetDrawOutcome();
      return;
    }
    play_phase_ = PlayPhase::kDraw;
    return;
  }
}

void XueliuErmaState::ApplyAfterAddKongAction(Action action) {
  int responder = current_player_;

  if (action == kHuAction) {
    RecordPublicActionEvent(responder, action);
    hand_[responder][pending_kong_tile_]++;
    WinContext context;
    context.qiang_gang_hu = true;
    context.jin_gou_hu = IsJinGouHu(responder);
    context.di_hu =
        (responder == 1 && discard_count_[1] == 0 && discard_count_[0] == 1);
    context.winning_tile = pending_kong_tile_;
    ScoreUp(responder, context);
    ClearPendingAddKong();
    phase_ = Phase::kGameOver;
    return;
  }

  SPIEL_CHECK_EQ(action, kPassAction);
  RecordPublicActionEvent(responder, action);
  bool upgraded = false;
  for (auto& meld : melds_[pending_kong_player_]) {
    if (meld.type == MeldType::kPong && meld.tile_type == pending_kong_tile_) {
      meld.type = MeldType::kMingGang;
      upgraded = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(upgraded);
  kong_fan_[pending_kong_player_] += 1;
  current_player_ = pending_kong_player_;
  ClearPendingAddKong();
  is_gang_draw_ = true;
  if (wall_pos_ >= static_cast<int>(wall_.size())) {
    SetDrawOutcome();
    return;
  }
  play_phase_ = PlayPhase::kAfterKong;
}

void XueliuErmaState::ClearPendingAddKong() {
  pending_add_kong_ = false;
  pending_kong_player_ = kInvalidPlayer;
  pending_kong_tile_ = -1;
}

void XueliuErmaState::ClearDiscardContext() {
  last_discard_ = -1;
  last_discard_player_ = kInvalidPlayer;
  last_discard_after_kong_ = false;
}

// ---------- XueliuErmaGame ----------

XueliuErmaGame::XueliuErmaGame(const GameParameters& params)
    : Game(kGameType, params) {}

std::vector<int> XueliuErmaGame::ObservationTensorShape() const {
  return {kObservationTensorSize};
}

std::vector<int> XueliuErmaGame::InformationStateTensorShape() const {
  return {kInformationStateTensorSize};
}

}  // namespace xueliu_erma
}  // namespace open_spiel
