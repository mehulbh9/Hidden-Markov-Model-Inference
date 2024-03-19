import numpy as np
import graphics
import rover
from typing import List, Tuple
from collections import defaultdict


def compute_list_forward(hidden_states, observed_states, prior_distribution, trans_model, obs_model, observations):
    """
    Computes the forward messages for each time step of the model.
    """
    total_obs = len(observations)
    prob_observed_on_state = 0

    # Use a list comprehension to create the list_forward list
    list_forward = [rover.Distribution({}) for _ in range(total_obs)]
    list_forward[0] = rover.Distribution({})

    starting_point = observations[0]
    
    if starting_point is None:
        prob_observed_on_state = 1


    # Compute initial forward message for first time step
    for state in hidden_states:
        if prob_observed_on_state != 1:
            prob_observed_on_state = obs_model(state)[starting_point]
        prior_state_prob = prior_distribution[state]

        # Add to forward message if non-zero probability
        if prob_observed_on_state != 0 and prior_state_prob != 0:
            list_forward[0][state] = prob_observed_on_state * prior_state_prob

    list_forward[0].renormalize()

    # Compute forward messages for remaining time steps
    for i in range(1, total_obs):
        list_forward[i] = rover.Distribution({})
        observed_state = observations[i]

    # Compute forward message for each possible hidden state
    i = 1
    while i < total_obs:
        list_forward[i] = rover.Distribution({})
        observed_state = observations[i]

        # Compute forward message for each possible hidden state
        for state in hidden_states:
            if observed_state is None:
                prob_observed_on_state = 1
            else:
                prob_observed_on_state = obs_model(state)[observed_state]

            total_prob = 0

            # Compute total probability of reaching current state from previous state
            for prev_state in list_forward[i - 1]:
                total_prob += list_forward[i - 1][prev_state] * trans_model(prev_state)[state]

            # Add to forward message if non-zero probability
            if (prob_observed_on_state * total_prob) != 0:
                list_forward[i][state] = prob_observed_on_state * total_prob

        list_forward[i].renormalize()
        i += 1


    return list_forward




def compute_backward_messages(hidden_states: List, observed_states: List, 
                              trans_model: callable, obs_model: callable, 
                              observations: List) -> List:
    """
    Computes the backward messages for each time step of the model.

    Args:
        hidden_states: A list of all possible hidden states.
        observed_states: A list of all possible observed states.
        trans_model: A function that takes in a hidden state and returns a
            distribution over possible next states.
        obs_model: A function that takes in a hidden state and returns a
            distribution over possible observed states.
        observations: A list of observed states.

    Returns:
        A list of backward messages, where each message is a Distribution object
        representing the probabilities of the hidden states at each time step,
        conditioned on all future observations.
    """
    num_timesteps = len(observations)
    backward_messages = [None] * num_timesteps
    backward_messages[num_timesteps - 1] = defaultdict(lambda: 1)
    
    # Iterate backwards through time and compute backward messages
    for t in reversed(range(num_timesteps - 1)):
        backward_messages[t] = defaultdict(float)
        for hidden_state in hidden_states:
            # Compute the total probability of all possible next states
            # given the current hidden state and the observation at time t+1
            total_prob = sum(backward_messages[t+1][next_state] * obs_model(next_state)[observations[t+1]] * trans_prob
                             for next_state, trans_prob in trans_model(hidden_state).items())
            if total_prob != 0:
                # Update the probability of the current hidden state at time t
                # conditioned on all future observations
                backward_messages[t][hidden_state] = total_prob / sum(trans_model(hidden_state).values())
    
    # Convert the defaultdicts to Distribution objects
    return [rover.Distribution(dict(d)) for d in backward_messages]



def compute_marginals(list_forward: List, backward_messages: List, 
                      all_states: List) -> List:
    """
    Computes the marginal distributions at each time step using the forward
    and backward messages and a list of all possible states.

    Args:
    list_forward: A list of forward messages, where the i-th message
        is a distribution over the i-th state.
    backward_messages: A list of backward messages, where the i-th message
        is a distribution over the i-th state.
    all_states: A list of all possible states.

    Returns:
    A list of marginal distributions at each time step, with each distribution
    encoded as a Distribution object. The i-th Distribution corresponds to
    time step i.
    """
    marginals = []

    # For each time step
    for t, fwd_msg in enumerate(list_forward):
        bwd_msg = backward_messages[t]

        # Compute the marginal distribution for this time step
        marginal = rover.Distribution({
            state: fwd_msg[state] * bwd_msg[state]
            for state in fwd_msg if bwd_msg[state] != 0
        })
        marginals.append(marginal)

    return marginals




def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states
    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)
    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """
    
    list_forward = compute_list_forward(all_possible_hidden_states, all_possible_observed_states,
                                                prior_distribution, transition_model, observation_model, observations)
    backward_messages = compute_backward_messages(all_possible_hidden_states, all_possible_observed_states,
                                                transition_model, observation_model, observations)
    marginals = compute_marginals(list_forward, backward_messages, all_possible_hidden_states)

    return marginals



def initialize_viterbi(observations: List, 
                       all_possible_hidden_states: List, 
                       prior_distribution: dict, 
                       observation_model: callable) -> List:
    """
    Initializes the Viterbi algorithm with the prior probabilities and 
    the probability of observing the first observation for each hidden state.
    
    Args:
    observations: A list of observed states.
    all_possible_hidden_states: A list of all possible hidden states.
    prior_distribution: A dictionary with prior probabilities for each hidden state.
    observation_model: A function that takes in a hidden state and returns a
        distribution over possible observed states.

    Returns:
    A list of Distribution objects representing the log-probabilities of each 
    hidden state at each time step.
    """
    w = [rover.Distribution() for _ in range(len(observations))]
    initial_observed_position = observations[0]
    for z0 in all_possible_hidden_states:
        if initial_observed_position == None:
            initial_prob_position_on_state = 1
        else:
            initial_prob_position_on_state = observation_model(z0)[initial_observed_position]
        prior_z0 = prior_distribution[z0]
        if (initial_prob_position_on_state != 0) and (prior_z0 != 0):
            w[0][z0] = np.log(initial_prob_position_on_state) + np.log(prior_z0)
    return w


def forward_pass_viterbi(observations: List, 
                         all_possible_hidden_states: List, 
                         w: List, 
                         transition_model: callable, 
                         observation_model) -> Tuple[List, List]:
    """
    Computes the forward messages for each time step of the Viterbi algorithm.

    Args:
    observations: A list of observed states.
    all_possible_hidden_states: A list of all possible hidden states.
    w: A list of Distribution objects representing the log-probabilities of each 
        hidden state at each time step.
    transition_model: A function that takes in a hidden state and returns a
        distribution over possible next states.
    observation_model: A function that takes in a hidden state and returns a
        distribution over possible observed states.

    Returns:
    A tuple containing:
    - A list of Distribution objects representing the log-probabilities of each 
        hidden state at each time step.
    - A list of dictionaries representing the most likely previous hidden state 
        at each time step.
    """
    z_previous = [None] * len(observations)
    for i in range(1, len(observations)):
        w[i] = rover.Distribution({})
        z_previous[i] = dict()
        observed_position = observations[i]
        for zi in all_possible_hidden_states:
            if observed_position == None:
                prob_position_on_state = 1
            else:
                prob_position_on_state = observation_model(zi)[observed_position]
            max_term = -np.inf
            for zi_minus_1 in w[i-1]:
                if transition_model(zi_minus_1)[zi] != 0:
                    potential_max_term = np.log(transition_model(zi_minus_1)[zi]) + w[i-1][zi_minus_1]
                    if (potential_max_term > max_term) and (prob_position_on_state != 0):
                        max_term = potential_max_term
                        z_previous[i][zi] = zi_minus_1 # keep track of which zi_minus_1 can maximize w[i][zi]

            if prob_position_on_state != 0:
                w[i][zi] = np.log(prob_position_on_state) + max_term
    return w, z_previous


def backward_pass_viterbi(w: List[dict], z_previous: List[dict], 
                          all_states: List) -> List:
    """
    Computes the most likely sequence of hidden states using the Viterbi algorithm.

    Args:
        w: A list of Distribution objects representing the log-probabilities
            of each hidden state at each time step, obtained from the forward pass.
        z_previous: A list of dictionaries representing the most likely previous
            hidden state at each time step, obtained from the forward pass.
        all_states: A list of all possible hidden states.

    Returns:
        A list of the most likely sequence of hidden states.
    """
    T = len(w)
    estimated_hidden_states = [None] * T
    max_w = -np.inf

    # Find the hidden state with the highest log-probability at the last time step
    for state in w[-1]:
        potential_max_w = w[-1][state]
        if potential_max_w > max_w:
            max_w = potential_max_w
            estimated_hidden_states[-1] = state

    # Backtrack to find the most likely sequence of hidden states
    for i in range(1, T):
        estimated_hidden_states[T-1-i] = z_previous[T-i][estimated_hidden_states[T-i]]

    return estimated_hidden_states



def Viterbi(all_possible_hidden_states, all_possible_observed_states,
            prior_distribution, transition_model, observation_model, observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.
    Output
    ------
    A list of estimated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """
    
    w = initialize_viterbi(observations, all_possible_hidden_states, prior_distribution, observation_model)
    w, z_previous = forward_pass_viterbi(observations, all_possible_hidden_states, w, transition_model, observation_model)
    estimated_hidden_states = backward_pass_viterbi(w, z_previous, all_possible_hidden_states)
    return estimated_hidden_states



if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
