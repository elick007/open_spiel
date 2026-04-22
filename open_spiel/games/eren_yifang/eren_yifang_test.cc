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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace eren_yifang {
namespace {

void BasicGameTests() {
  testing::LoadGameTest("eren_yifang");
  testing::RandomSimTest(*LoadGame("eren_yifang"), 50);
}

void ObservationFeatureTests() {
  std::shared_ptr<const Game> game = LoadGame("eren_yifang");
  SPIEL_CHECK_TRUE(game != nullptr);

  const std::vector<int> observation_shape = game->ObservationTensorShape();
  SPIEL_CHECK_EQ(observation_shape.size(), 3);
  SPIEL_CHECK_EQ(observation_shape[0], kObservationChannels);
  SPIEL_CHECK_EQ(observation_shape[1], kObservationHeight);
  SPIEL_CHECK_EQ(observation_shape[2], kObservationWidth);

  const std::vector<int> information_state_shape =
      game->InformationStateTensorShape();
  SPIEL_CHECK_EQ(information_state_shape.size(), 1);
  SPIEL_CHECK_EQ(information_state_shape[0], kInformationStateTensorSize);

  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->ChanceOutcomes()[0].first);
  }

  const std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(legal_actions.empty());
  const Action first_action = legal_actions[0];
  state->ApplyAction(first_action);

  const std::vector<float> obs0 = state->ObservationTensor(0);
  const std::vector<float> obs1 = state->ObservationTensor(1);
  const std::vector<float> info0 = state->InformationStateTensor(0);
  const std::vector<float> info1 = state->InformationStateTensor(1);
  const int position_offset = 0;
  const int self_last_action_offset = position_offset + kPositionFeatureChannels;
  const int opp_last_action_offset = self_last_action_offset + kNumDistinctActions;

  SPIEL_CHECK_EQ(static_cast<int>(obs0.size()), kObservationTensorSize);
  SPIEL_CHECK_EQ(static_cast<int>(obs1.size()), kObservationTensorSize);
  SPIEL_CHECK_EQ(static_cast<int>(info0.size()), kInformationStateTensorSize);
  SPIEL_CHECK_EQ(static_cast<int>(info1.size()), kInformationStateTensorSize);
  SPIEL_CHECK_EQ(info0[position_offset + 0], 1.0f);
  SPIEL_CHECK_EQ(info0[position_offset + 1], 0.0f);
  SPIEL_CHECK_EQ(info1[position_offset + 0], 0.0f);
  SPIEL_CHECK_EQ(info1[position_offset + 1], 1.0f);

  SPIEL_CHECK_EQ(info0[self_last_action_offset + first_action], 1.0f);
  SPIEL_CHECK_EQ(info1[opp_last_action_offset + first_action], 1.0f);
}

}  // namespace
}  // namespace eren_yifang
}  // namespace open_spiel

int main() {
  open_spiel::eren_yifang::BasicGameTests();
  open_spiel::eren_yifang::ObservationFeatureTests();
}
