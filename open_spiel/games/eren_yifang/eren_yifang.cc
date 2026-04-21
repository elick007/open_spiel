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

#include "open_spiel/games/eren_yifang/eren_yifang.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace eren_yifang {
namespace {

const GameType kGameType{
    /*short_name=*/"eren_yifang",
    /*long_name=*/"Er Ren Yi Fang",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ErenYifangGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

int FirstNonZero(const std::array<int, kNumTileTypes>& counts) {
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    if (counts[tile] > 0) {
      return tile;
    }
  }
  return -1;
}

bool CanFormSets(std::array<int, kNumTileTypes>& counts, int sets_left) {
  if (sets_left == 0) {
    return std::accumulate(counts.begin(), counts.end(), 0) == 0;
  }
  const int first = FirstNonZero(counts);
  if (first == -1) {
    return false;
  }

  if (counts[first] >= 3) {
    counts[first] -= 3;
    if (CanFormSets(counts, sets_left - 1)) {
      counts[first] += 3;
      return true;
    }
    counts[first] += 3;
  }

  if (first + 2 < kNumTileTypes && counts[first] > 0 && counts[first + 1] > 0 &&
      counts[first + 2] > 0) {
    counts[first]--;
    counts[first + 1]--;
    counts[first + 2]--;
    if (CanFormSets(counts, sets_left - 1)) {
      counts[first]++;
      counts[first + 1]++;
      counts[first + 2]++;
      return true;
    }
    counts[first]++;
    counts[first + 1]++;
    counts[first + 2]++;
  }

  return false;
}

bool CanFormTripletSets(std::array<int, kNumTileTypes>& counts, int sets_left) {
  if (sets_left == 0) {
    return std::accumulate(counts.begin(), counts.end(), 0) == 0;
  }
  const int first = FirstNonZero(counts);
  if (first == -1 || counts[first] < 3) {
    return false;
  }
  counts[first] -= 3;
  const bool valid = CanFormTripletSets(counts, sets_left - 1);
  counts[first] += 3;
  return valid;
}

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

std::string ErenYifangState::TileTypeToString(int tile_type) const {
  return absl::StrCat(tile_type + 1, "W");
}

int ErenYifangState::CountConcealedTiles(int player) const {
  return std::accumulate(hand_[player].begin(), hand_[player].end(), 0);
}

int ErenYifangState::CountPhysicalTiles(int player, int tile_type) const {
  int count = hand_[player][tile_type];
  for (const Meld& meld : melds_[player]) {
    if (meld.tile_type != tile_type) {
      continue;
    }
    count += (meld.type == MeldType::kPong) ? 3 : 4;
  }
  return count;
}

int ErenYifangState::RequiredConcealedSetCount(int player) const {
  return kWinningSetCount - static_cast<int>(melds_[player].size());
}

bool ErenYifangState::HasWallTiles() const {
  return wall_pos_ < static_cast<int>(wall_.size());
}

int ErenYifangState::CurrentNodeType() const {
  if (phase_ == Phase::kDeal) {
    return 0;
  }
  if (phase_ == Phase::kGameOver) {
    return 6;
  }
  switch (play_phase_) {
    case PlayPhase::kDraw:
      return 1;
    case PlayPhase::kAfterDraw:
      return 2;
    case PlayPhase::kAfterPong:
      return 3;
    case PlayPhase::kAfterDiscard:
      return 4;
    case PlayPhase::kAfterAddKong:
      return 5;
  }
  SpielFatalError("Unknown node type.");
}

void ErenYifangState::WriteObservationFeatures(Player player,
                                               absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
  std::fill(values.begin(), values.end(), 0.0f);

  const int opponent = 1 - player;

  std::array<int, kNumTileTypes> self_exposed{};
  std::array<int, kNumTileTypes> self_concealed_kongs{};
  for (const Meld& meld : melds_[player]) {
    if (meld.type == MeldType::kAnGang) {
      self_concealed_kongs[meld.tile_type] += kTilesPerKind;
    } else {
      self_exposed[meld.tile_type] +=
          (meld.type == MeldType::kPong) ? 3 : kTilesPerKind;
    }
  }

  std::array<int, kNumTileTypes> opponent_exposed{};
  int opponent_concealed_kong_count = 0;
  for (const Meld& meld : melds_[opponent]) {
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
  // Appendix-B-style public/private layout adapted to this ruleset.
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

  // Side features keep only information observable to this player.
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
  side[side_offset++] = last_draw_was_kong_replacement_ ? 1.0f : 0.0f;

  for (int p = 0; p < kNumPlayers; ++p) {
    side[side_offset++] = static_cast<float>(score_adjustments_[p]);
  }
  side[side_offset++] =
      static_cast<float>(wall_.size() - wall_pos_) / kWallSize;
  SPIEL_CHECK_EQ(side_offset, kSideFeatureSize);
}

void ErenYifangState::RecordPublicActionEvent(Player player, Action action) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  last_action_ = action;
}

bool ErenYifangState::IsWinningHand(
    const std::array<int, kNumTileTypes>& hand, int meld_count) const {
  const int required_sets = kWinningSetCount - meld_count;
  if (required_sets < 0) {
    return false;
  }
  const int total_tiles = std::accumulate(hand.begin(), hand.end(), 0);
  if (total_tiles != 3 * required_sets + 2) {
    return false;
  }

  for (int pair_tile = 0; pair_tile < kNumTileTypes; ++pair_tile) {
    if (hand[pair_tile] < 2) {
      continue;
    }
    std::array<int, kNumTileTypes> remaining = hand;
    remaining[pair_tile] -= 2;
    if (CanFormSets(remaining, required_sets)) {
      return true;
    }
  }
  return false;
}

bool ErenYifangState::CanHu(int player) const {
  return IsWinningHand(hand_[player], static_cast<int>(melds_[player].size()));
}

bool ErenYifangState::CanHuWithTile(int player, int tile_type) const {
  std::array<int, kNumTileTypes> test_hand = hand_[player];
  test_hand[tile_type]++;
  return IsWinningHand(test_hand, static_cast<int>(melds_[player].size()));
}

bool ErenYifangState::CanHuWithAnyTile(int player) const {
  for (int tile_type = 0; tile_type < kNumTileTypes; ++tile_type) {
    if (CanHuWithTile(player, tile_type)) {
      return true;
    }
  }
  return false;
}

bool ErenYifangState::IsAllTriplets(int player) const {
  const int required_sets = RequiredConcealedSetCount(player);
  if (required_sets < 0) {
    return false;
  }

  for (int pair_tile = 0; pair_tile < kNumTileTypes; ++pair_tile) {
    if (hand_[player][pair_tile] < 2) {
      continue;
    }
    std::array<int, kNumTileTypes> remaining = hand_[player];
    remaining[pair_tile] -= 2;
    if (CanFormTripletSets(remaining, required_sets)) {
      return true;
    }
  }
  return false;
}

bool ErenYifangState::IsMenQing(int player) const {
  for (const Meld& meld : melds_[player]) {
    if (meld.type != MeldType::kAnGang) {
      return false;
    }
  }
  return true;
}

bool ErenYifangState::IsZhongZhang(int player) const {
  return CountPhysicalTiles(player, 0) == 0 &&
         CountPhysicalTiles(player, kNumTileTypes - 1) == 0;
}

bool ErenYifangState::IsJinGouHu(int player) const {
  return static_cast<int>(melds_[player].size()) == kWinningSetCount &&
         CountConcealedTiles(player) == 2;
}

bool ErenYifangState::IsJiaXinFive(int player, int winning_tile) const {
  if (winning_tile != 4) {
    return false;
  }

  const int required_sets = RequiredConcealedSetCount(player);
  if (required_sets <= 0 || hand_[player][3] <= 0 || hand_[player][4] <= 0 ||
      hand_[player][5] <= 0) {
    return false;
  }

  std::array<int, kNumTileTypes> remaining = hand_[player];
  remaining[3]--;
  remaining[4]--;
  remaining[5]--;

  const int remaining_tiles =
      std::accumulate(remaining.begin(), remaining.end(), 0);
  if (remaining_tiles != 3 * (required_sets - 1) + 2) {
    return false;
  }

  for (int pair_tile = 0; pair_tile < kNumTileTypes; ++pair_tile) {
    if (remaining[pair_tile] < 2) {
      continue;
    }
    std::array<int, kNumTileTypes> pair_removed = remaining;
    pair_removed[pair_tile] -= 2;
    if (CanFormSets(pair_removed, required_sets - 1)) {
      return true;
    }
  }

  return false;
}

int ErenYifangState::CountRoots(int player) const {
  int roots = 0;
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    if (CountPhysicalTiles(player, tile) == kTilesPerKind) {
      ++roots;
    }
  }
  return roots;
}

int ErenYifangState::BaseFan(int player) const {
  int fan = 0;
  if (IsAllTriplets(player)) {
    fan = fan + 2;
  }
  if (IsMenQing(player)) {
    ++fan;
  }
  if (IsZhongZhang(player)) {
    ++fan;
  }
  return fan;
}

int ErenYifangState::BonusFan(int player, const WinContext& context) const {
  int bonus = CountRoots(player);
  if (context.gang_shang_kai_hua) {
    ++bonus;
  }
  if (context.gang_shang_pao) {
    ++bonus;
  }
  if (context.qiang_gang_hu) {
    ++bonus;
  }
  if (context.saodi_hu) {
    ++bonus;
  }
  if (context.jin_gou_hu) {
    ++bonus;
  }
  if (context.haidi_pao) {
    ++bonus;
  }
  if (context.tian_hu) {
    bonus += 3;
  }
  if (context.di_hu) {
    bonus += 2;
  }
  if (IsJiaXinFive(player, context.winning_tile)) {
    ++bonus;
  }
  if (context.self_draw) {
    ++bonus;
  }

  return bonus;
}

void ErenYifangState::ApplyScoreDelta(int player, int points) {
  const int opponent = 1 - player;
  score_adjustments_[player] += points;
  score_adjustments_[opponent] -= points;
}

void ErenYifangState::SetRollbackableKong(int player, int points) {
  rollbackable_kong_active_ = true;
  rollbackable_kong_player_ = player;
  rollbackable_kong_points_ = points;
  last_draw_was_kong_replacement_ = true;
}

void ErenYifangState::ClearRollbackableKong() {
  rollbackable_kong_active_ = false;
  rollbackable_kong_player_ = kInvalidPlayer;
  rollbackable_kong_points_ = 0;
}

void ErenYifangState::UndoRollbackableKong() {
  if (!rollbackable_kong_active_) {
    return;
  }
  ApplyScoreDelta(rollbackable_kong_player_, -rollbackable_kong_points_);
  ClearRollbackableKong();
}

void ErenYifangState::ClearPendingAddKong() {
  pending_add_kong_ = false;
  pending_kong_player_ = kInvalidPlayer;
  pending_kong_tile_ = -1;
}

void ErenYifangState::ClearDiscardContext() {
  last_discard_ = -1;
  last_discard_player_ = kInvalidPlayer;
  last_discard_after_kong_ = false;
}

void ErenYifangState::SetDrawOutcome() {
  returns_[0] = 0.0;
  returns_[1] = 0.0;
  const bool player0_ready = CanHuWithAnyTile(0);
  const bool player1_ready = CanHuWithAnyTile(1);
  if (player0_ready != player1_ready) {
    returns_[player0_ready ? 0 : 1] = 2.0;
    returns_[player0_ready ? 1 : 0] = -2.0;
  }
  phase_ = Phase::kGameOver;
}

void ErenYifangState::ScoreUp(int winner, const WinContext& context) {
  const int loser = 1 - winner;
  const int total_fan = std::min(BaseFan(winner) + BonusFan(winner, context), 4);
  double hu_score = std::ldexp(1.0, total_fan);

  returns_[0] = score_adjustments_[0];
  returns_[1] = score_adjustments_[1];
  returns_[winner] += hu_score;
  returns_[loser] -= hu_score;
  phase_ = Phase::kGameOver;
}

ErenYifangState::ErenYifangState(std::shared_ptr<const Game> game)
    : State(game) {
  wall_.reserve(kNumTiles);
  for (int tile_type = 0; tile_type < kNumTileTypes; ++tile_type) {
    for (int copy = 0; copy < kTilesPerKind; ++copy) {
      wall_.push_back(tile_type);
    }
  }
}

Player ErenYifangState::CurrentPlayer() const {
  if (phase_ == Phase::kGameOver) {
    return kTerminalPlayerId;
  }
  if (phase_ == Phase::kDeal || play_phase_ == PlayPhase::kDraw) {
    return kChancePlayerId;
  }
  return current_player_;
}

std::string ErenYifangState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Draw ", TileTypeToString(wall_[action]));
  }
  if (action >= kDiscardActionBase && action < kPongAction) {
    return absl::StrCat("Discard ", TileTypeToString(action));
  }
  if (action == kPongAction) {
    return "Pong";
  }
  if (action >= kKongActionBase && action < kHuAction) {
    return absl::StrCat("Kong ", TileTypeToString(action - kKongActionBase));
  }
  if (action == kHuAction) {
    return "Hu";
  }
  if (action == kPassAction) {
    return "Pass";
  }
  return absl::StrCat("Unknown(", action, ")");
}

std::string ErenYifangState::ToString() const {
  std::string out;
  for (int player = 0; player < kNumPlayers; ++player) {
    absl::StrAppend(&out, "Player ", player, " hand: ");
    for (int tile = 0; tile < kNumTileTypes; ++tile) {
      for (int count = 0; count < hand_[player][tile]; ++count) {
        absl::StrAppend(&out, TileTypeToString(tile), " ");
      }
    }
    if (!melds_[player].empty()) {
      absl::StrAppend(&out, "| Melds: ");
      for (const Meld& meld : melds_[player]) {
        const char* label = "";
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
        absl::StrAppend(&out, "[", label, " ", TileTypeToString(meld.tile_type),
                        "] ");
      }
    }
    absl::StrAppend(&out, "\n");
  }
  absl::StrAppend(&out, "Wall remaining: ", wall_.size() - wall_pos_, "\n");
  if (last_discard_ >= 0) {
    absl::StrAppend(&out, "Last discard: ", TileTypeToString(last_discard_),
                    " by P", last_discard_player_, "\n");
  }
  if (pending_add_kong_) {
    absl::StrAppend(&out, "Pending add-kong: P", pending_kong_player_, " ",
                    TileTypeToString(pending_kong_tile_), "\n");
  }
  if (IsTerminal()) {
    absl::StrAppend(&out, "Returns: P0=", returns_[0], " P1=", returns_[1],
                    "\n");
  }
  return out;
}

bool ErenYifangState::IsTerminal() const { return phase_ == Phase::kGameOver; }

std::vector<double> ErenYifangState::Returns() const { return returns_; }

std::string ErenYifangState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  return ObservationString(player);
}

void ErenYifangState::InformationStateTensor(Player player,
                                             absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_EQ(values.size(), kInformationStateTensorSize);
  WriteObservationFeatures(player, values);
}

std::string ErenYifangState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  const int opponent = 1 - player;
  std::string out = absl::StrFormat("P%d hand:", player);
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    for (int count = 0; count < hand_[player][tile]; ++count) {
      absl::StrAppend(&out, " ", TileTypeToString(tile));
    }
  }

  bool has_self_public_meld = false;
  int opponent_concealed_kongs = 0;
  for (const Meld& meld : melds_[player]) {
    if (!has_self_public_meld) {
      absl::StrAppend(&out, " | Self melds:");
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
    absl::StrAppend(&out, " [", label, " ", TileTypeToString(meld.tile_type),
                    "]");
  }

  if (!discard_history_[player].empty()) {
    absl::StrAppend(&out, " | Self discards:");
    for (int tile_type : discard_history_[player]) {
      absl::StrAppend(&out, " ", TileTypeToString(tile_type));
    }
  }

  bool has_opponent_exposed_meld = false;
  for (const Meld& meld : melds_[opponent]) {
    if (meld.type == MeldType::kAnGang) {
      ++opponent_concealed_kongs;
      continue;
    }
    if (!has_opponent_exposed_meld) {
      absl::StrAppend(&out, " | Opp exposed:");
      has_opponent_exposed_meld = true;
    }
    const char* label = meld.type == MeldType::kPong ? "Pong" : "MingGang";
    absl::StrAppend(&out, " [", label, " ", TileTypeToString(meld.tile_type),
                    "]");
  }
  if (opponent_concealed_kongs > 0) {
    absl::StrAppend(&out, " | Opp an-gang count:", opponent_concealed_kongs);
  }
  if (!discard_history_[opponent].empty()) {
    absl::StrAppend(&out, " | Opp discards:");
    for (int tile_type : discard_history_[opponent]) {
      absl::StrAppend(&out, " ", TileTypeToString(tile_type));
    }
  }

  absl::StrAppend(&out, " | Wall: ", wall_.size() - wall_pos_);
  if (last_discard_ >= 0) {
    absl::StrAppend(&out, " | Last discard: ", TileTypeToString(last_discard_),
                    " by P", last_discard_player_);
  }
  if (last_action_ >= 0 && last_action_ < kNumDistinctActions) {
    absl::StrAppend(&out, " | Last action: ",
                    ActionToString(/*player=*/0, last_action_));
  }
  return out;
}

void ErenYifangState::ObservationTensor(Player player,
                                        absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
  WriteObservationFeatures(player, values);
}

std::vector<std::pair<Action, double>> ErenYifangState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  if (phase_ == Phase::kDeal) {
    const int remaining = kNumTiles - tiles_dealt_;
    const double probability = 1.0 / remaining;
    for (int action = tiles_dealt_; action < kNumTiles; ++action) {
      outcomes.emplace_back(action, probability);
    }
    return outcomes;
  }

  const int remaining = static_cast<int>(wall_.size()) - wall_pos_;
  SPIEL_CHECK_GT(remaining, 0);
  const double probability = 1.0 / remaining;
  for (int action = wall_pos_; action < static_cast<int>(wall_.size());
       ++action) {
    outcomes.emplace_back(action, probability);
  }
  return outcomes;
}

std::unique_ptr<State> ErenYifangState::Clone() const {
  return std::make_unique<ErenYifangState>(*this);
}

std::vector<Action> ErenYifangState::DealLegalActions() const {
  std::vector<Action> actions;
  actions.reserve(kNumTiles - tiles_dealt_);
  for (int action = tiles_dealt_; action < kNumTiles; ++action) {
    actions.push_back(action);
  }
  return actions;
}

std::vector<Action> ErenYifangState::DrawLegalActions() const {
  std::vector<Action> actions;
  actions.reserve(wall_.size() - wall_pos_);
  for (int action = wall_pos_; action < static_cast<int>(wall_.size());
       ++action) {
    actions.push_back(action);
  }
  return actions;
}

std::vector<Action> ErenYifangState::AfterDrawLegalActions() const {
  std::vector<Action> actions;
  if (CanHu(current_player_)) {
    actions.push_back(kHuAction);
  }

  if (static_cast<int>(melds_[current_player_].size()) < kWinningSetCount) {
    for (int tile = 0; tile < kNumTileTypes; ++tile) {
      if (hand_[current_player_][tile] == 4) {
        actions.push_back(kKongActionBase + tile);
      }
    }
  }

  for (const Meld& meld : melds_[current_player_]) {
    if (meld.type == MeldType::kPong &&
        hand_[current_player_][meld.tile_type] > 0) {
      actions.push_back(kKongActionBase + meld.tile_type);
    }
  }

  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    if (hand_[current_player_][tile] > 0) {
      actions.push_back(kDiscardActionBase + tile);
    }
  }

  std::sort(actions.begin(), actions.end());
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
  return actions;
}

std::vector<Action> ErenYifangState::AfterPongLegalActions() const {
  std::vector<Action> actions;
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    if (hand_[current_player_][tile] > 0) {
      actions.push_back(kDiscardActionBase + tile);
    }
  }
  return actions;
}

std::vector<Action> ErenYifangState::AfterDiscardLegalActions() const {
  std::vector<Action> actions;
  const int responder = current_player_;
  if (CanHuWithTile(responder, last_discard_)) {
    actions.push_back(kHuAction);
  }
  if (static_cast<int>(melds_[responder].size()) < kWinningSetCount) {
    if (hand_[responder][last_discard_] >= 2) {
      actions.push_back(kPongAction);
    }
    if (hand_[responder][last_discard_] >= 3) {
      actions.push_back(kKongActionBase + last_discard_);
    }
  }
  actions.push_back(kPassAction);
  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> ErenYifangState::AfterAddKongLegalActions() const {
  std::vector<Action> actions;
  if (CanHuWithTile(current_player_, pending_kong_tile_)) {
    actions.push_back(kHuAction);
  }
  actions.push_back(kPassAction);
  return actions;
}

std::vector<Action> ErenYifangState::LegalActions() const {
  switch (phase_) {
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kPlay:
      switch (play_phase_) {
        case PlayPhase::kDraw:
          return DrawLegalActions();
        case PlayPhase::kAfterDraw:
          return AfterDrawLegalActions();
        case PlayPhase::kAfterPong:
          return AfterPongLegalActions();
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

void ErenYifangState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      ApplyDealAction(action);
      return;
    case Phase::kPlay:
      switch (play_phase_) {
        case PlayPhase::kDraw:
          ApplyDrawAction(action);
          return;
        case PlayPhase::kAfterDraw:
          ApplyAfterDrawAction(action);
          return;
        case PlayPhase::kAfterPong:
          ApplyAfterPongAction(action);
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
      SpielFatalError("Cannot act in terminal states.");
  }
}

void ErenYifangState::ApplyDealAction(Action action) {
  SPIEL_CHECK_GE(action, tiles_dealt_);
  SPIEL_CHECK_LT(action, kNumTiles);

  std::swap(wall_[tiles_dealt_], wall_[action]);
  const int tile_type = wall_[tiles_dealt_];
  const int player = (tiles_dealt_ % kNumPlayers == 0) ? 0 : 1;
  hand_[player][tile_type]++;
  ++tiles_dealt_;

  if (tiles_dealt_ == kInitialDealCount) {
    phase_ = Phase::kPlay;
    play_phase_ = PlayPhase::kAfterDraw;
    current_player_ = 0;
    wall_pos_ = tiles_dealt_;
  }
}

void ErenYifangState::ApplyDrawAction(Action action) {
  SPIEL_CHECK_GE(action, wall_pos_);
  SPIEL_CHECK_LT(action, static_cast<int>(wall_.size()));

  std::swap(wall_[wall_pos_], wall_[action]);
  const int tile_type = wall_[wall_pos_];
  ++wall_pos_;
  hand_[current_player_][tile_type]++;
  last_drawn_tile_ = tile_type;
  play_phase_ = PlayPhase::kAfterDraw;
}

void ErenYifangState::ApplyAfterDrawAction(Action action) {
  if (action == kHuAction) {
    RecordPublicActionEvent(current_player_, action);
    WinContext context;
    context.self_draw = true;
    context.gang_shang_kai_hua = last_draw_was_kong_replacement_;
    context.saodi_hu = (wall_pos_ == static_cast<int>(wall_.size()));
    context.jin_gou_hu = IsJinGouHu(current_player_);
    context.tian_hu = (current_player_ == 0 && num_discards_total_ == 0);
    context.di_hu =
        (current_player_ == 1 && discard_count_[1] == 0 && discard_count_[0] == 1);
    context.winning_tile = last_drawn_tile_;
    ScoreUp(current_player_, context);
    return;
  }

  if (action >= kKongActionBase && action < kHuAction) {
    const int tile_type = action - kKongActionBase;
    RecordPublicActionEvent(current_player_, action);

    if (hand_[current_player_][tile_type] == 4 &&
        static_cast<int>(melds_[current_player_].size()) < kWinningSetCount) {
      hand_[current_player_][tile_type] = 0;
      melds_[current_player_].push_back({MeldType::kAnGang, tile_type});
      ApplyScoreDelta(current_player_, 2);
      SetRollbackableKong(current_player_, 2);
      if (!HasWallTiles()) {
        SetDrawOutcome();
        return;
      }
      play_phase_ = PlayPhase::kDraw;
      return;
    }

    SPIEL_CHECK_GT(hand_[current_player_][tile_type], 0);
    hand_[current_player_][tile_type]--;
    pending_add_kong_ = true;
    pending_kong_player_ = current_player_;
    pending_kong_tile_ = tile_type;
    current_player_ = 1 - current_player_;
    play_phase_ = PlayPhase::kAfterAddKong;
    return;
  }

  SPIEL_CHECK_GE(action, kDiscardActionBase);
  SPIEL_CHECK_LT(action, kPongAction);
  const int tile_type = action - kDiscardActionBase;
  SPIEL_CHECK_GT(hand_[current_player_][tile_type], 0);
  RecordPublicActionEvent(current_player_, action);
  hand_[current_player_][tile_type]--;
  last_discard_ = tile_type;
  last_discard_player_ = current_player_;
  last_discard_after_kong_ = rollbackable_kong_active_;
  discard_history_[current_player_].push_back(tile_type);
  ++discard_count_[current_player_];
  ++num_discards_total_;
  current_player_ = 1 - current_player_;
  play_phase_ = PlayPhase::kAfterDiscard;
  last_draw_was_kong_replacement_ = false;
}

void ErenYifangState::ApplyAfterPongAction(Action action) {
  SPIEL_CHECK_GE(action, kDiscardActionBase);
  SPIEL_CHECK_LT(action, kPongAction);
  const int tile_type = action - kDiscardActionBase;
  SPIEL_CHECK_GT(hand_[current_player_][tile_type], 0);
  RecordPublicActionEvent(current_player_, action);
  hand_[current_player_][tile_type]--;
  last_discard_ = tile_type;
  last_discard_player_ = current_player_;
  last_discard_after_kong_ = false;
  discard_history_[current_player_].push_back(tile_type);
  ++discard_count_[current_player_];
  ++num_discards_total_;
  current_player_ = 1 - current_player_;
  play_phase_ = PlayPhase::kAfterDiscard;
  last_draw_was_kong_replacement_ = false;
}

void ErenYifangState::ApplyAfterDiscardAction(Action action) {
  const int responder = current_player_;

  if (action == kHuAction) {
    RecordPublicActionEvent(responder, action);
    hand_[responder][last_discard_]++;
    WinContext context;
    context.gang_shang_pao = last_discard_after_kong_;
    context.haidi_pao = (wall_pos_ == static_cast<int>(wall_.size()));
    context.jin_gou_hu = IsJinGouHu(responder);
    context.di_hu =
        (responder == 1 && discard_count_[1] == 0 && discard_count_[0] == 1);
    context.winning_tile = last_discard_;
    if (last_discard_after_kong_) {
      UndoRollbackableKong();
    }
    ScoreUp(responder, context);
    return;
  }

  if (action == kPongAction) {
    RecordPublicActionEvent(responder, action);
    SPIEL_CHECK_GE(hand_[responder][last_discard_], 2);
    hand_[responder][last_discard_] -= 2;
    melds_[responder].push_back({MeldType::kPong, last_discard_});
    ClearRollbackableKong();
    ClearDiscardContext();
    current_player_ = responder;
    play_phase_ = PlayPhase::kAfterPong;
    return;
  }

  if (action >= kKongActionBase && action < kHuAction) {
    RecordPublicActionEvent(responder, action);
    const int tile_type = action - kKongActionBase;
    SPIEL_CHECK_EQ(tile_type, last_discard_);
    SPIEL_CHECK_GE(hand_[responder][tile_type], 3);
    hand_[responder][tile_type] -= 3;
    melds_[responder].push_back({MeldType::kMingGang, tile_type});
    ApplyScoreDelta(responder, 2);
    SetRollbackableKong(responder, 2);
    ClearDiscardContext();
    current_player_ = responder;
    if (!HasWallTiles()) {
      SetDrawOutcome();
      return;
    }
    play_phase_ = PlayPhase::kDraw;
    return;
  }

  SPIEL_CHECK_EQ(action, kPassAction);
  RecordPublicActionEvent(responder, action);
  ClearRollbackableKong();
  ClearDiscardContext();
  if (!HasWallTiles()) {
    SetDrawOutcome();
    return;
  }
  play_phase_ = PlayPhase::kDraw;
  last_draw_was_kong_replacement_ = false;
}

void ErenYifangState::ApplyAfterAddKongAction(Action action) {
  const int responder = current_player_;

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
    return;
  }

  SPIEL_CHECK_EQ(action, kPassAction);
  RecordPublicActionEvent(responder, action);
  const int kong_player = pending_kong_player_;
  bool upgraded = false;
  for (Meld& meld : melds_[kong_player]) {
    if (meld.type == MeldType::kPong && meld.tile_type == pending_kong_tile_) {
      meld.type = MeldType::kMingGang;
      upgraded = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(upgraded);
  ApplyScoreDelta(kong_player, 1);
  SetRollbackableKong(kong_player, 1);
  ClearPendingAddKong();
  current_player_ = kong_player;
  if (!HasWallTiles()) {
    SetDrawOutcome();
    return;
  }
  play_phase_ = PlayPhase::kDraw;
}

ErenYifangGame::ErenYifangGame(const GameParameters& params)
    : Game(kGameType, params) {}

std::vector<int> ErenYifangGame::InformationStateTensorShape() const {
  return {kInformationStateTensorSize};
}

std::vector<int> ErenYifangGame::ObservationTensorShape() const {
  return {kObservationTensorSize};
}

}  // namespace eren_yifang
}  // namespace open_spiel
