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
#include "open_spiel/utils/tensor_view.h"

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

bool InActionRange(Action action, int base, int end) {
  return action >= base && action <= end;
}

int TileFromAction(Action action, int base) { return action - base; }

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
    --counts[first];
    --counts[first + 1];
    --counts[first + 2];
    if (CanFormSets(counts, sets_left - 1)) {
      ++counts[first];
      ++counts[first + 1];
      ++counts[first + 2];
      return true;
    }
    ++counts[first];
    ++counts[first + 1];
    ++counts[first + 2];
  }

  return false;
}

void WriteTileCountPlane(const std::array<int, kNumTileTypes>& counts,
                         absl::Span<float> values) {
  SPIEL_CHECK_EQ(values.size(), kObservationHeight * kObservationWidth);
  std::fill(values.begin(), values.end(), 0.0f);
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    const int copies = std::min(counts[tile], kObservationWidth);
    for (int copy = 0; copy < copies; ++copy) {
      values[tile * kObservationWidth + copy] = 1.0f;
    }
  }
}

}  // namespace

ErenYifangState::ErenYifangState(std::shared_ptr<const Game> game)
    : State(game) {
  wall_.reserve(kNumTiles);
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    for (int copy = 0; copy < kTilesPerKind; ++copy) {
      wall_.push_back(tile);
    }
  }
  for (auto& per_player : gong_mode_) {
    per_player.fill(-1);
  }
}

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

bool ErenYifangState::HasWallTiles() const {
  return wall_pos_ < static_cast<int>(wall_.size());
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
  ++test_hand[tile_type];
  return IsWinningHand(test_hand,
                       static_cast<int>(melds_[player].size()));
}

bool ErenYifangState::IsDuiDuiHu(int player) const {
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    if (hand_[player][tile] > 0 && hand_[player][tile] < 2) {
      return false;
    }
  }
  return true;
}

bool ErenYifangState::IsMenQing(int player) const {
  for (const Meld& meld : melds_[player]) {
    if (meld.type != MeldType::kConcealedGong) {
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

  const int required_sets = kWinningSetCount -
                            static_cast<int>(melds_[player].size());
  if (required_sets <= 0 || hand_[player][3] <= 0 || hand_[player][4] <= 0 ||
      hand_[player][5] <= 0) {
    return false;
  }

  std::array<int, kNumTileTypes> remaining = hand_[player];
  --remaining[3];
  --remaining[4];
  --remaining[5];

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
  if (IsDuiDuiHu(player)) {
    fan += 2;
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
  int fan = CountRoots(player);
  if (context.gang_shang_kai_hua) {
    ++fan;
  }
  if (context.qiang_gang_hu) {
    ++fan;
  }
  if (context.gang_shang_pao) {
    ++fan;
  }
  if (context.saodi_hu) {
    ++fan;
  }
  if (context.haidi_pao) {
    ++fan;
  }
  if (context.jin_gou_hu) {
    ++fan;
  }
  if (context.tian_hu) {
    fan += 3;
  }
  if (context.di_hu) {
    fan += 2;
  }
  if (context.self_draw) {
    ++fan;
  }
  if (IsJiaXinFive(player, context.winning_tile)) {
    ++fan;
  }
  return fan;
}

int ErenYifangState::KongScore(int player) const {
  int score = 0;
  for (const Meld& meld : melds_[player]) {
    switch (meld.type) {
      case MeldType::kPong:
        break;
      case MeldType::kDirectGong:
      case MeldType::kConcealedGong:
        score += 2;
        break;
      case MeldType::kAddGong:
        score += 1;
        break;
    }
  }
  return score;
}

int ErenYifangState::FirstConcealedGongTile(int player) const {
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    if (hand_[player][tile] == kTilesPerKind) {
      return tile;
    }
  }
  return -1;
}

int ErenYifangState::FirstAddGongTile(int player) const {
  for (const Meld& meld : melds_[player]) {
    if (meld.type == MeldType::kPong &&
        hand_[player][meld.tile_type] > 0) {
      return meld.tile_type;
    }
  }
  return -1;
}

void ErenYifangState::WriteObservationFeatures(Player player,
                                               absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), kObservationTensorSize);
  std::fill(values.begin(), values.end(), 0.0f);

  std::array<int, kNumTileTypes> table_counts{};
  std::array<std::array<int, kNumTileTypes>, kNumPlayers> pile_counts{};
  for (int tile : table_) {
    ++table_counts[tile];
  }
  for (int meld_player = 0; meld_player < kNumPlayers; ++meld_player) {
    for (const Meld& meld : melds_[meld_player]) {
      pile_counts[meld_player][meld.tile_type] +=
          (meld.type == MeldType::kPong) ? 3 : 4;
    }
  }

  int offset = 0;
  const int plane_size = kObservationHeight * kObservationWidth;
  auto write_plane = [&](const std::array<int, kNumTileTypes>& counts) {
    WriteTileCountPlane(counts, values.subspan(offset, plane_size));
    offset += plane_size;
  };

  write_plane(hand_[player]);
  write_plane(table_counts);
  write_plane(pile_counts[0]);
  write_plane(pile_counts[1]);

  for (int seat = 0; seat < kNumPlayers; ++seat) {
    absl::Span<float> plane = values.subspan(offset, plane_size);
    std::fill(plane.begin(), plane.end(), seat == player ? 1.0f : 0.0f);
    offset += plane_size;
  }

  SPIEL_CHECK_EQ(offset, kObservationTensorSize);
}

void ErenYifangState::WriteInformationStateFeatures(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), kInformationStateTensorSize);
  WriteObservationFeatures(player, values);
}

void ErenYifangState::RecordPublicActionEvent(Action action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);
  last_action_ = action;
}

void ErenYifangState::ClearDiscardContext() {
  last_discard_ = -1;
  last_discard_player_ = kInvalidPlayer;
}

void ErenYifangState::ClearPendingAddGong() {
  pending_add_gong_ = false;
  pending_kong_player_ = kInvalidPlayer;
  pending_kong_tile_ = -1;
}

void ErenYifangState::UndoPendingAddGong() {
  if (!pending_add_gong_) {
    return;
  }
  bool reverted = false;
  for (Meld& meld : melds_[pending_kong_player_]) {
    if (meld.type == MeldType::kAddGong &&
        meld.tile_type == pending_kong_tile_) {
      meld.type = MeldType::kPong;
      reverted = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(reverted);
  gong_mode_[pending_kong_player_][pending_kong_tile_] = -1;
  ClearPendingAddGong();
}

void ErenYifangState::EnterDrawChance(Player player) {
  current_player_ = player;
  play_phase_ = PlayPhase::kDrawChance;
  if (!HasWallTiles()) {
    SetDrawOutcome();
  }
}

void ErenYifangState::SetDrawOutcome() {
  returns_[0] = 0.0;
  returns_[1] = 0.0;
  phase_ = Phase::kGameOver;
}

void ErenYifangState::ScoreUp(int winner, const WinContext& context) {
  const int loser = 1 - winner;
  const int total_fan =
      std::min(BaseFan(winner) + BonusFan(winner, context), 4);
  const double base_score = std::ldexp(1.0, total_fan);
  const double winner_score = base_score + KongScore(winner);
  const double loser_score = KongScore(loser);
  const double score = std::abs(winner_score - loser_score);

  returns_[winner] = score;
  returns_[loser] = -score;
  phase_ = Phase::kGameOver;
}

Player ErenYifangState::CurrentPlayer() const {
  if (phase_ == Phase::kGameOver) {
    return kTerminalPlayerId;
  }
  if (phase_ == Phase::kDeal ||
      (phase_ == Phase::kPlay && play_phase_ == PlayPhase::kDrawChance)) {
    return kChancePlayerId;
  }
  return current_player_;
}

std::string ErenYifangState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat(
        phase_ == Phase::kDeal ? "Deal " : "ChanceDraw ",
        TileTypeToString(wall_[action]), " @", action);
  }
  if (action == kDrawAction) {
    return "Draw";
  }
  if (action == kPongAction) {
    return "Pong";
  }
  if (action == kGongAction) {
    return "Gong";
  }
  if (action == kStandAction) {
    return "Stand";
  }
  if (action == kHuAction) {
    return "Hu";
  }
  if (action == kZimoAction) {
    return "Zimo";
  }
  if (InActionRange(action, kDiscardActionBase, kDiscardActionEnd)) {
    return absl::StrCat("Discard ",
                        TileTypeToString(TileFromAction(action,
                                                        kDiscardActionBase)));
  }
  return absl::StrCat("Unknown(", action, ")");
}

std::string ErenYifangState::ToString() const {
  std::string out;
  for (int player = 0; player < kNumPlayers; ++player) {
    absl::StrAppend(&out, "Player ", player, " hand:");
    for (int tile = 0; tile < kNumTileTypes; ++tile) {
      for (int count = 0; count < hand_[player][tile]; ++count) {
        absl::StrAppend(&out, " ", TileTypeToString(tile));
      }
    }
    if (!melds_[player].empty()) {
      absl::StrAppend(&out, " | melds:");
      for (const Meld& meld : melds_[player]) {
        const char* label = "";
        switch (meld.type) {
          case MeldType::kPong:
            label = "Pong";
            break;
          case MeldType::kDirectGong:
            label = "DirectGong";
            break;
          case MeldType::kAddGong:
            label = "AddGong";
            break;
          case MeldType::kConcealedGong:
            label = "ConcealedGong";
            break;
        }
        absl::StrAppend(&out, " [", label, " ",
                        TileTypeToString(meld.tile_type), "]");
      }
    }
    if (!discard_history_[player].empty()) {
      absl::StrAppend(&out, " | discards:");
      for (int tile : discard_history_[player]) {
        absl::StrAppend(&out, " ", TileTypeToString(tile));
      }
    }
    absl::StrAppend(&out, "\n");
  }
  absl::StrAppend(&out, "Wall remaining: ",
                  static_cast<int>(wall_.size()) - wall_pos_, "\n");
  if (phase_ == Phase::kPlay) {
    absl::StrAppend(&out, "Current player: ", current_player_, "\n");
  }
  if (last_discard_ >= 0) {
    absl::StrAppend(&out, "Last discard: ", TileTypeToString(last_discard_),
                    " by P", last_discard_player_, "\n");
  }
  if (pending_add_gong_) {
    absl::StrAppend(&out, "Pending rob-gong check: P", pending_kong_player_,
                    " ", TileTypeToString(pending_kong_tile_), "\n");
  }
  if (last_action_ != kInvalidAction) {
    absl::StrAppend(&out, "Last action: ",
                    ActionToString(/*player=*/0, last_action_), "\n");
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
  WriteInformationStateFeatures(player, values);
}

std::string ErenYifangState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, kNumPlayers);
  const int opponent = 1 - player;
  std::string out = absl::StrCat("P", player, " hand:");
  for (int tile = 0; tile < kNumTileTypes; ++tile) {
    for (int count = 0; count < hand_[player][tile]; ++count) {
      absl::StrAppend(&out, " ", TileTypeToString(tile));
    }
  }

  bool has_self_exposed = false;
  bool has_self_hidden = false;
  for (const Meld& meld : melds_[player]) {
    if (meld.type == MeldType::kConcealedGong) {
      if (!has_self_hidden) {
        absl::StrAppend(&out, " | Self hidden:");
        has_self_hidden = true;
      }
      absl::StrAppend(&out, " [ConcealedGong ",
                      TileTypeToString(meld.tile_type), "]");
    } else {
      if (!has_self_exposed) {
        absl::StrAppend(&out, " | Self exposed:");
        has_self_exposed = true;
      }
      const char* label = nullptr;
      switch (meld.type) {
        case MeldType::kPong:
          label = "Pong";
          break;
        case MeldType::kDirectGong:
          label = "Gong";
          break;
        case MeldType::kAddGong:
          label = "AddGong";
          break;
        case MeldType::kConcealedGong:
          label = "ConcealedGong";
          break;
      }
      absl::StrAppend(&out, " [", label, " ",
                      TileTypeToString(meld.tile_type), "]");
    }
  }

  if (!discard_history_[player].empty()) {
    absl::StrAppend(&out, " | Self discards:");
    for (int tile : discard_history_[player]) {
      absl::StrAppend(&out, " ", TileTypeToString(tile));
    }
  }

  bool has_opp_exposed = false;
  bool has_opp_hidden = false;
  for (const Meld& meld : melds_[opponent]) {
    if (meld.type == MeldType::kConcealedGong) {
      if (!has_opp_hidden) {
        absl::StrAppend(&out, " | Opp hidden:");
        has_opp_hidden = true;
      }
      absl::StrAppend(&out, " [ConcealedGong ",
                      TileTypeToString(meld.tile_type), "]");
    } else {
      if (!has_opp_exposed) {
        absl::StrAppend(&out, " | Opp exposed:");
        has_opp_exposed = true;
      }
      const char* label = nullptr;
      switch (meld.type) {
        case MeldType::kPong:
          label = "Pong";
          break;
        case MeldType::kDirectGong:
          label = "Gong";
          break;
        case MeldType::kAddGong:
          label = "AddGong";
          break;
        case MeldType::kConcealedGong:
          label = "ConcealedGong";
          break;
      }
      absl::StrAppend(&out, " [", label, " ",
                      TileTypeToString(meld.tile_type), "]");
    }
  }

  if (!discard_history_[opponent].empty()) {
    absl::StrAppend(&out, " | Opp discards:");
    for (int tile : discard_history_[opponent]) {
      absl::StrAppend(&out, " ", TileTypeToString(tile));
    }
  }

  absl::StrAppend(&out, " | Wall: ", wall_.size() - wall_pos_);
  if (last_discard_ >= 0) {
    absl::StrAppend(&out, " | Last discard: ", TileTypeToString(last_discard_),
                    " by P", last_discard_player_);
  }
  if (last_action_ != kInvalidAction) {
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
    const double prob = 1.0 / remaining;
    for (int action = tiles_dealt_; action < kNumTiles; ++action) {
      outcomes.emplace_back(action, prob);
    }
    return outcomes;
  }

  SPIEL_CHECK_EQ(static_cast<int>(phase_), static_cast<int>(Phase::kPlay));
  SPIEL_CHECK_EQ(static_cast<int>(play_phase_),
                 static_cast<int>(PlayPhase::kDrawChance));
  const int remaining = static_cast<int>(wall_.size()) - wall_pos_;
  SPIEL_CHECK_GT(remaining, 0);
  const double prob = 1.0 / remaining;
  for (int action = wall_pos_; action < static_cast<int>(wall_.size());
       ++action) {
    outcomes.emplace_back(action, prob);
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

std::vector<Action> ErenYifangState::ActorTurnLegalActions() const {
  std::vector<Action> actions;
  if (CanHu(current_player_)) {
    actions.push_back(kZimoAction);
  }

  if (HasWallTiles()) {
    const bool can_concealed_gong =
        FirstConcealedGongTile(current_player_) >= 0;
    const bool can_add_gong =
        current_player_ != last_player_ &&
        FirstAddGongTile(current_player_) >= 0;
    if (can_concealed_gong || can_add_gong) {
      actions.push_back(kGongAction);
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

std::vector<Action> ErenYifangState::RespondToDiscardLegalActions() const {
  std::vector<Action> actions;

  if (CanHuWithTile(current_player_, last_discard_)) {
    actions.push_back(kHuAction);
  }
  if (HasWallTiles()) {
    if (hand_[current_player_][last_discard_] >= 3) {
      actions.push_back(kGongAction);
    }
    if (hand_[current_player_][last_discard_] >= 2) {
      actions.push_back(kPongAction);
    }
    if (!actions.empty()) {
      actions.push_back(kDrawAction);
    }
  }

  std::sort(actions.begin(), actions.end());
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
  return actions;
}

std::vector<Action> ErenYifangState::RespondToAddGongLegalActions() const {
  std::vector<Action> actions;
  if (CanHuWithTile(current_player_, pending_kong_tile_)) {
    actions.push_back(kHuAction);
  }
  return actions;
}

std::vector<Action> ErenYifangState::LegalActions() const {
  switch (phase_) {
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kPlay:
      switch (play_phase_) {
        case PlayPhase::kActorTurn:
          return ActorTurnLegalActions();
        case PlayPhase::kRespondToDiscard:
          return RespondToDiscardLegalActions();
        case PlayPhase::kRespondToAddGong:
          return RespondToAddGongLegalActions();
        case PlayPhase::kDrawChance: {
          std::vector<Action> actions;
          actions.reserve(static_cast<int>(wall_.size()) - wall_pos_);
          for (int action = wall_pos_; action < static_cast<int>(wall_.size());
               ++action) {
            actions.push_back(action);
          }
          return actions;
        }
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
        case PlayPhase::kActorTurn:
          ApplyActorTurnAction(action);
          return;
        case PlayPhase::kRespondToDiscard:
          ApplyRespondToDiscardAction(action);
          return;
        case PlayPhase::kRespondToAddGong:
          ApplyRespondToAddGongAction(action);
          return;
        case PlayPhase::kDrawChance:
          ApplyDrawChanceAction(action);
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
  const int tile = wall_[tiles_dealt_];

  int player = 0;
  if (tiles_dealt_ >= kNonDealerInitialHandSize &&
      tiles_dealt_ < 2 * kNonDealerInitialHandSize) {
    player = 1;
  }
  hand_[player][tile]++;
  if (tiles_dealt_ == kInitialDealCount - 1) {
    last_drawn_tile_ = tile;
  }
  ++tiles_dealt_;

  if (tiles_dealt_ == kInitialDealCount) {
    phase_ = Phase::kPlay;
    play_phase_ = PlayPhase::kActorTurn;
    current_player_ = 0;
    wall_pos_ = tiles_dealt_;
  }
}

void ErenYifangState::ApplyDrawChanceAction(Action action) {
  SPIEL_CHECK_GE(action, wall_pos_);
  SPIEL_CHECK_LT(action, static_cast<int>(wall_.size()));

  std::swap(wall_[wall_pos_], wall_[action]);
  const int tile = wall_[wall_pos_];
  ++wall_pos_;
  hand_[current_player_][tile]++;
  last_drawn_tile_ = tile;
  play_phase_ = PlayPhase::kActorTurn;
}

void ErenYifangState::ApplyActorTurnAction(Action action) {
  const Player player = current_player_;

  if (action == kZimoAction) {
    RecordPublicActionEvent(action);
    WinContext context;
    context.self_draw = true;
    context.gang_shang_kai_hua = is_gonging_[player];
    context.saodi_hu = !HasWallTiles();
    context.jin_gou_hu = IsJinGouHu(player);
    context.tian_hu = is_first_action_[player] && player == 0;
    context.di_hu = is_first_action_[player] && player == 1;
    context.winning_tile = last_drawn_tile_;
    ScoreUp(player, context);
    return;
  }

  if (action == kGongAction) {
    SPIEL_CHECK_TRUE(HasWallTiles());
    RecordPublicActionEvent(action);
    is_first_action_[player] = false;

    const int add_gong_tile =
        current_player_ != last_player_ ? FirstAddGongTile(player) : -1;
    if (add_gong_tile >= 0) {
      bool upgraded = false;
      for (Meld& meld : melds_[player]) {
        if (meld.type == MeldType::kPong && meld.tile_type == add_gong_tile) {
          meld.type = MeldType::kAddGong;
          upgraded = true;
          break;
        }
      }
      SPIEL_CHECK_TRUE(upgraded);
      --hand_[player][add_gong_tile];
      gong_mode_[player][add_gong_tile] = 2;
      pending_add_gong_ = true;
      pending_kong_player_ = player;
      pending_kong_tile_ = add_gong_tile;
      is_gonging_[player] = true;
      discard_after_gong_[player] = false;

      const Player responder = 1 - player;
      if (CanHuWithTile(responder, add_gong_tile)) {
        current_player_ = responder;
        play_phase_ = PlayPhase::kRespondToAddGong;
      } else {
        ClearPendingAddGong();
        EnterDrawChance(player);
      }
      return;
    }

    const int concealed_gong_tile = FirstConcealedGongTile(player);
    SPIEL_CHECK_GE(concealed_gong_tile, 0);
    hand_[player][concealed_gong_tile] = 0;
    melds_[player].push_back(
        {MeldType::kConcealedGong, concealed_gong_tile});
    gong_mode_[player][concealed_gong_tile] = 0;
    is_gonging_[player] = true;
    discard_after_gong_[player] = false;
    EnterDrawChance(player);
    return;
  }

  SPIEL_CHECK_TRUE(InActionRange(action, kDiscardActionBase,
                                 kDiscardActionEnd));
  const int tile = TileFromAction(action, kDiscardActionBase);
  SPIEL_CHECK_GT(hand_[player][tile], 0);

  RecordPublicActionEvent(action);
  is_first_action_[player] = false;
  --hand_[player][tile];
  discard_history_[player].push_back(tile);
  table_.push_back(tile);
  last_discard_ = tile;
  last_discard_player_ = player;
  last_player_ = player;
  discard_after_gong_[player] = is_gonging_[player];
  is_gonging_[player] = false;

  const Player responder = 1 - player;
  const bool can_respond =
      CanHuWithTile(responder, tile) ||
      (HasWallTiles() && hand_[responder][tile] >= 2);
  if (can_respond) {
    current_player_ = responder;
    play_phase_ = PlayPhase::kRespondToDiscard;
  } else if (HasWallTiles()) {
    EnterDrawChance(responder);
  } else {
    SetDrawOutcome();
  }
}

void ErenYifangState::ApplyRespondToDiscardAction(Action action) {
  const Player responder = current_player_;
  SPIEL_CHECK_GE(last_discard_, 0);

  if (action == kDrawAction) {
    SPIEL_CHECK_TRUE(HasWallTiles());
    RecordPublicActionEvent(action);
    ClearDiscardContext();
    EnterDrawChance(responder);
    return;
  }

  if (action == kHuAction) {
    RecordPublicActionEvent(action);
    SPIEL_CHECK_FALSE(table_.empty());
    SPIEL_CHECK_EQ(table_.back(), last_discard_);
    table_.pop_back();
    ++hand_[responder][last_discard_];
    WinContext context;
    context.self_draw = false;
    context.gang_shang_pao = discard_after_gong_[last_discard_player_];
    context.haidi_pao = !HasWallTiles();
    context.jin_gou_hu = IsJinGouHu(responder);
    context.tian_hu = is_first_action_[responder] && responder == 0;
    context.di_hu = is_first_action_[responder] && responder == 1;
    context.winning_tile = last_discard_;
    ScoreUp(responder, context);
    return;
  }

  if (action == kPongAction) {
    const int tile = last_discard_;
    SPIEL_CHECK_GE(hand_[responder][tile], 2);
    RecordPublicActionEvent(action);
    is_first_action_[responder] = false;
    hand_[responder][tile] -= 2;
    melds_[responder].push_back({MeldType::kPong, tile});
    SPIEL_CHECK_FALSE(table_.empty());
    SPIEL_CHECK_EQ(table_.back(), tile);
    table_.pop_back();
    ClearDiscardContext();
    last_player_ = responder;
    current_player_ = responder;
    play_phase_ = PlayPhase::kActorTurn;
    return;
  }

  SPIEL_CHECK_EQ(action, kGongAction);
  SPIEL_CHECK_TRUE(HasWallTiles());
  const int tile = last_discard_;
  SPIEL_CHECK_GE(hand_[responder][tile], 3);
  RecordPublicActionEvent(action);
  is_first_action_[responder] = false;
  hand_[responder][tile] -= 3;
  melds_[responder].push_back({MeldType::kDirectGong, tile});
  gong_mode_[responder][tile] = 1;
  SPIEL_CHECK_FALSE(table_.empty());
  SPIEL_CHECK_EQ(table_.back(), tile);
  table_.pop_back();
  ClearDiscardContext();
  is_gonging_[responder] = true;
  discard_after_gong_[responder] = false;
  EnterDrawChance(responder);
}

void ErenYifangState::ApplyRespondToAddGongAction(Action action) {
  const Player responder = current_player_;
  SPIEL_CHECK_TRUE(pending_add_gong_);

  if (action == kHuAction) {
    const int winning_tile = pending_kong_tile_;
    RecordPublicActionEvent(action);
    ++hand_[responder][winning_tile];
    UndoPendingAddGong();
    WinContext context;
    context.self_draw = false;
    context.qiang_gang_hu = true;
    context.jin_gou_hu = IsJinGouHu(responder);
    context.tian_hu = is_first_action_[responder] && responder == 0;
    context.di_hu = is_first_action_[responder] && responder == 1;
    context.winning_tile = winning_tile;
    ScoreUp(responder, context);
    return;
  }

  SpielFatalError(
      "RLCard mahjong_two_by_one only exposes Hu when robbing a gong.");
}

ErenYifangGame::ErenYifangGame(const GameParameters& params)
    : Game(kGameType, params) {}

std::vector<int> ErenYifangGame::InformationStateTensorShape() const {
  return {kObservationChannels, kObservationHeight, kObservationWidth};
}

std::vector<int> ErenYifangGame::ObservationTensorShape() const {
  return {kObservationChannels, kObservationHeight, kObservationWidth};
}

}  // namespace eren_yifang
}  // namespace open_spiel
