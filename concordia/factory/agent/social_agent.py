"""A factory implementing a thinking agent."""

from collections.abc import Callable
import datetime
import json

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.utils import measurements as measurements_lib
import numpy as np

DEFAULT_PLANNING_HORIZON = (  # update long term goal
    'both immediate actions and long-term implications'
)
DEFAULT_GOAL_COMPONENT_NAME = (  # TODO: update to improve the finding of goal
    'Goal'
)


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build a social agent with decision making
  Aiming for balancing risk/reward and long/short term goals
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError(
        'This function is meant for a main character '
        'but it was called on a supporting character.'
    )

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(
      memory
  )  # TODO: Might change to an updated memeory bank later on

  measurements = measurements_lib.Measurements()

  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  observation_label = (  # update prompt
      '\nDetailed Observation Including Social Context'
  )
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )

  observation_summary_label = (  # Changed to make sure encapsulate more information
      '\nComprehensive Summary of Events, Relationships & Patterns'
  )
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(
          hours=48
      ),  # Extended from 24-> 48
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )

  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',  # TODO: Change prefix prompt
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )

  # Enhanced identity with social dimensions
  identity_label = '\nIdentity characteristics (Personal & Social Dimensions)'
  identity_characteristics = agent_components.question_of_query_associated_memories.IdentityWithoutPreAct(
      model=model,
      logging_channel=measurements.get_channel('IdentityWithoutPreAct').on_next,
      pre_act_key=identity_label,
  )

  self_perception_label = (  # KEY PART: Updated the thinking of agents to be deeper
      f'\nQuestion: How does {agent_name} understand themselves and their role'
      ' in their social network? Consider:\n- Core values, beliefs, and'
      ' aspirations\n- Relationship patterns and social dynamics\n-'
      ' Communication style and preferences\n- Past experiences that shape'
      ' current behavior\nAnalysis'
  )
  self_perception = agent_components.question_of_recent_memories.SelfPerception(
      model=model,
      components={_get_class_name(identity_characteristics): identity_label},
      pre_act_key=self_perception_label,
      logging_channel=measurements.get_channel('SelfPerception').on_next,
  )

  situation_perception_label = (  # KEY PART: Updated the thinking of agents to be deeper
      f"\nQuestion: What is {agent_name}'s current situation and its"
      ' implications? Consider:\n- Immediate circumstances and social'
      ' context\n- Relevant relationships and their dynamics\n- Potential'
      ' risks and opportunities\n- Historical patterns and their relevance\n-'
      ' Long-term implications of the situation\nAnalysis'
  )
  situation_perception = (
      agent_components.question_of_recent_memories.SituationPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(observation_summary): observation_summary_label,
          },
          clock_now=clock.now,
          pre_act_key=situation_perception_label,
          logging_channel=measurements.get_channel(
              'SituationPerception'
          ).on_next,
      )
  )

  person_by_situation_label = (
      f'\nDecision Analysis for {agent_name}:\n'
      '1. Social Context Assessment:\n'
      '   - Current relationship dynamics\n'
      '   - Social expectations and norms\n'
      '   - Communication implications\n'
      '2. Risk-Reward Analysis:\n'
      '   - Potential benefits and opportunities\n'
      '   - Possible risks and challenges\n'
      '   - Impact on relationships and reputation\n'
      '3. Timeline Considerations:\n'
      '   - Immediate consequences\n'
      '   - Medium-term implications\n'  # wanted to expand just from short term and long term
      '   - Long-term effects\n'
      '4. Strategic Alignment:\n'
      '   - Consistency with personal values\n'
      '   - Fit with long-term goals\n'
      '   - Balance of social and personal interests\n'
      'Based on this analysis, what actions would be most appropriate?\n'
      'Analysis'
  )
  person_by_situation = (
      agent_components.question_of_recent_memories.PersonBySituation(
          model=model,
          components={
              _get_class_name(self_perception): self_perception_label,
              _get_class_name(situation_perception): situation_perception_label,
          },
          clock_now=clock.now,
          pre_act_key=person_by_situation_label,
          logging_channel=measurements.get_channel('PersonBySituation').on_next,
      )
  )

  relevant_memories_label = '\nRelevant Experiences, Relationships & Patterns'
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is',
      },
      num_memories_to_retrieve=15,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )

  plan_components = {}
  if config.goal:
    goal_label = (  # TODO: Updated working, but unsure about this one
        '\nStrategic Goals & Objectives'
    )
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(
            DEFAULT_GOAL_COMPONENT_NAME
        ).on_next,
    )
    plan_components[DEFAULT_GOAL_COMPONENT_NAME] = goal_label
  else:
    overarching_goal = None

  plan_components.update({
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(self_perception): self_perception_label,
      _get_class_name(situation_perception): situation_perception_label,
      _get_class_name(person_by_situation): person_by_situation_label,
  })

  plan = agent_components.plan.Plan(
      model=model,
      observation_component_name=_get_class_name(observation),
      components=plan_components,
      clock_now=clock.now,
      goal_component_name=_get_class_name(person_by_situation),
      horizon=DEFAULT_PLANNING_HORIZON,
      pre_act_key='\nStrategic Action Plan',
      logging_channel=measurements.get_channel('Plan').on_next,
  )

  entity_components = (
      instructions,
      observation,
      observation_summary,
      relevant_memories,
      self_perception,
      situation_perception,
      person_by_situation,
      plan,
      time_display,
      # Components that do not provide pre_act context.
      identity_characteristics,
  )

  components_of_agent = {
      _get_class_name(component): component for component in entity_components
  }
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
  ] = agent_components.memory_component.MemoryComponent(raw_memory)

  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[DEFAULT_GOAL_COMPONENT_NAME] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, DEFAULT_GOAL_COMPONENT_NAME)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
      config=config,
  )

  return agent


def save_to_json(
    agent: entity_agent_with_logging.EntityAgentWithLogging,
) -> str:
  """Saves an agent to JSON data.

  This function saves the agent's state to a JSON string, which can be loaded
  afterwards with `rebuild_from_json`. The JSON data
  includes the state of the agent's context components, act component, memory,
  agent name and the initial config. The clock, model and embedder are not
  saved and will have to be provided when the agent is rebuilt. The agent must
  be in the `READY` phase to be saved.

  Args:
    agent: The agent to save.

  Returns:
    A JSON string representing the agent's state.

  Raises:
    ValueError: If the agent is not in the READY phase.
  """

  if agent.get_phase() != entity_component.Phase.READY:
    raise ValueError('The agent must be in the `READY` phase to be saved.')

  data = {
      component_name: agent.get_component(component_name).get_state()
      for component_name in agent.get_all_context_components()
  }

  data['act_component'] = agent.get_act_component().get_state()

  config = agent.get_config()
  if config is not None:
    data['agent_config'] = config.to_dict()

  return json.dumps(data)


def rebuild_from_json(
    json_data: str,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
    embedder: Callable[[str], np.ndarray],
    memory_importance: Callable[[str], float] | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Rebuilds an agent from JSON data."""

  data = json.loads(json_data)

  new_agent_memory = associative_memory.AssociativeMemory(
      sentence_embedder=embedder,
      importance=memory_importance,
      clock=clock.now,
      clock_step_size=clock.get_step_size(),
  )

  if 'agent_config' not in data:
    raise ValueError('The JSON data does not contain the agent config.')
  agent_config = formative_memories.AgentConfig.from_dict(
      data.pop('agent_config')
  )

  agent = build_agent(
      config=agent_config,
      model=model,
      memory=new_agent_memory,
      clock=clock,
  )

  for component_name in agent.get_all_context_components():
    agent.get_component(component_name).set_state(data.pop(component_name))

  agent.get_act_component().set_state(data.pop('act_component'))

  assert not data, f'Unused data {sorted(data)}'
  return agent
