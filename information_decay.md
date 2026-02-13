## Information Decay in the Environment

This document explains, **mathematically and intuitively**, how information decay is modeled in the simulation for:

- **`decay` mode** (continuous-time exponential disappearance of subjects)
- **`dynamic_pool` mode** (subjects appear and disappear with exponential lifetimes and exponential inter-arrival times)

The relevant implementation lives mainly in `environment.py` under the `Environment` class.

---

## Notation and shared concepts

- Let `t >= 0` denote **simulation time in seconds**, as returned by `Environment._elapsed_sim_seconds()`.
- Let `N0` be the **initial number of visible subject agents** at some reference time `t0`.
- Each subject agent carries a piece of information (a text snippet). When a subject becomes **invisible**, that information is no longer available to knowledge agents.
- The **exponential distribution with rate `lambda > 0`**:
  - Support: `x >= 0`
  - PDF: `f(x) = lambda * exp(-lambda * x)`
  - CDF: `F(x) = 1 - exp(-lambda * x)`
  - Survival: `P(X > x) = exp(-lambda * x)`
  - Mean: `E[X] = 1 / lambda`
  - Variance: `Var(X) = 1 / lambda^2`
  - Memoryless: `P(X > s + t | X > s) = P(X > t)`

In the code we usually parameterize the exponential distribution by its **mean**:

- Let `theta = lifetime_mean_time` (in seconds).  
  Then `lambda = 1 / theta`.

---

## 1. `decay` mode – Continuous-time exponential disappearance

### 1.1. High-level behavior

In **`decay` mode**, each subject agent is assigned an **independent exponential lifetime**. Once its lifetime expires, the subject becomes **permanently invisible** and does **not** reappear.

This models **one-shot information decay**: information (subjects) only disappears over time; it never returns.

### 1.2. Parameters

From `configs.yaml` under the active profile (e.g. `decay`):

- `information_teleportation.mode: decay`
- `information_teleportation.lifetime_mean_time = θ` (e.g. 10.0 seconds)
- Optional legacy parameters:
  - `decay_probability`
  - `interval_seconds`

In code (`environment.py`), these are read in the constructor:

- `self.lifetime_mean_time = lifetime_mean_time`
- `self.subject_lifetimes: Dict[SubjectAgent, float]` maps each subject to an **expiry time in seconds**.

### 1.3. Mathematical model

Fix a subject agent `i`.

1. At some reference time `t0` (when `decay` mode starts being applied to that agent), we draw a **lifetime**
   from an exponential distribution:

   - `Ti ~ Exp(lambda)` with `lambda = 1 / theta = 1 / lifetime_mean_time`.

2. We set the **expiry time**:

   - `Di = t0 + Ti`.

3. At any simulation time `t`, the subject is:
   - **visible** if `t < Di`,
   - **invisible** if `t >= Di`.

The survival probability that subject `i` is still visible at time `t >= t0` is:

- `P(subject i visible at time t)`
  `= P(Ti > t - t0)`
  `= exp(-lambda * (t - t0))`
  `= exp(-(t - t0) / theta)`.

If we have `N0` independent subjects at time `t0`, the **expected number visible** at time `t` is:

- `E[N(t)] = N0 * exp(-lambda * (t - t0)) = N0 * exp(-(t - t0) / theta)`.

Thus, in expectation, the **fraction** of information still visible decays exponentially with mean `theta`.

### 1.4. Implementation link

In `environment.py`, the logic is in `decay_subject_visibility`:

```298:363:environment.py
    def decay_subject_visibility(self):
        """
        Continuous-time exponential decay of subject visibility, similar in spirit
        to dynamic_pool but without reappearing snippets:

        - Each SUBJECT agent is assigned an independent exponential lifetime
          with rate λ = 1 / lifetime_mean_time (if provided).
        - When the current simulated time exceeds that agent's expiry time,
          the subject becomes permanently invisible.
        """
        subjects = [a for a in self._agents if getattr(a, "role", None) == "SUBJECT"]
        ...
        current_time = self._elapsed_sim_seconds()
        ...
        if self.lifetime_mean_time > 0:
            lambda_rate = 1.0 / float(self.lifetime_mean_time)
        ...
        # Initialize lifetimes for visible subjects without one
        for agent in subjects:
            if getattr(agent, "visible", True) and agent not in self.subject_lifetimes:
                lifetime = random.expovariate(lambda_rate)
                self.subject_lifetimes[agent] = current_time + lifetime
        ...
        # Hide subjects whose expiry time has passed
        for subject, expiry_time in list(self.subject_lifetimes.items()):
            if current_time >= expiry_time:
                subject.set_visible(False)
                del self.subject_lifetimes[subject]
```

The main loop calls this **every tick** when `mode == "decay"`:

```132:139:environment.py
            if self.teleportation_enabled:
                if self.teleportation_mode == "dynamic_pool":
                    self.dynamic_pool_update()
                elif self.teleportation_mode == "decay":
                    # Continuous-time exponential decay of subjects (no reappearance)
                    self.decay_subject_visibility()
                ...
```

### 1.5. Intuitive examples

#### Example A – Mean lifetime 10 seconds

- Suppose:
  - `theta = lifetime_mean_time = 10` seconds,
  - Initial subjects: `N0 = 20`,
  - Reference time `t0 = 0`.

Then:

- After **10 seconds**:
  - `E[N(10)] = 20 * exp(-10 / 10) = 20 * exp(-1) ≈ 20 * 0.3679 ≈ 7.36`.  
    So on average, **about 7–8 subjects** remain visible.

- After **20 seconds**:
  - `E[N(20)] = 20 * exp(-20 / 10) = 20 * exp(-2) ≈ 20 * 0.1353 ≈ 2.71`.  
    So on average, only **2–3 subjects** remain visible.

This exponential decay has the **memoryless** property: if we know that a subject has survived until time 10, the expected additional time it remains visible is still 10 seconds.

#### Example B – Interpreting theta

- If you set:
  - `lifetime_mean_time = 30.0`, you are saying:
    > “On average, each piece of information remains visible for **about 30 seconds**.”
  - `lifetime_mean_time = 5.0`, you are saying:
    > “On average, each piece of information disappears after **about 5 seconds**.”

You do **not** specify at which discrete time steps decay happens; the simulation time is continuous, and the code checks, at each tick, whether the subject’s **continuous exponential clock** has expired.

---

## 2. `dynamic_pool` mode – Exponential lifetimes **and** exponential arrivals

### 2.1. High-level behavior

In **`dynamic_pool` mode**, we model a **pool of inactive snippets** and a varying set of **active subjects**:

- Each active subject:
  - Carries a snippet (piece of information),
  - Has an **exponential lifetime** (like in `decay` mode),
  - When its lifetime expires, it becomes invisible and its snippet is returned to the **pool**.

- New subjects **appear** over time by sampling:
  - An **exponential inter-arrival time** between new subject appearances from the pool.

Thus, at any time, a random subset of snippets is active; snippets **churn**: they appear as subjects, disappear, and later may reappear as new subjects drawn from the pool.

### 2.2. Parameters

From the active profile (e.g. `dynamic_pool_experiment`) in `configs.yaml`:

- `information_teleportation.mode: dynamic_pool`
- `initial_active_count` – how many subjects are spawned initially.
- `appearance_mean_time = \alpha` – mean inter-arrival time between **new subjects appearing**.
- `lifetime_mean_time = \theta` – mean lifetime of each active subject.

In code (`environment.py`):

- `self.initial_active_count`
- `self.appearance_mean_time`
- `self.lifetime_mean_time`
- `self.snippet_pool` – list of snippets not currently active.
- `self.subject_lifetimes` – expiry times for currently active subjects.

### 2.3. Mathematical model

#### Lifetimes of active subjects

For each active subject `i`, at its activation time `t_start_i`, we draw:

- `T_life_i ~ Exp(lambda_life)` where `lambda_life = 1 / theta = 1 / lifetime_mean_time`.

The **disappearance time** is:

- `Di = t_start_i + T_life_i`.

The subject is visible for `t_start_i <= t < Di`, and becomes invisible for `t >= Di`, at which point:

- Its snippet is **returned to the pool**.
- The `SubjectAgent` instance is no longer considered an active information site.

The survival probability for a specific subject `i`, measured from its start time, is:

- `P(subject i visible at time t | t >= t_start_i)`
  `= exp(-lambda_life * (t - t_start_i))`
  `= exp(-(t - t_start_i) / theta)`.

#### Inter-arrival times of new subjects

New subject appearances from the pool are also modeled with an exponential distribution.

Let `Ak` be the **waiting time between the (k-1)-th and k-th subject appearance** from the pool:

- `Ak ~ Exp(lambda_arr)` where `lambda_arr = 1 / alpha = 1 / appearance_mean_time`.

If the last appearance happened at time `s`, then the next appearance time is:

- `T_next = s + Ak`.

Because the exponential is memoryless, the process of new subject arrivals is a **Poisson process in time** with rate `lambda_arr` (up to depletion of the finite snippet pool).

#### Pool dynamics

Let:

- `P(t)` = number of snippets currently **in the pool** (inactive),
- `A(t)` = number of **active subjects** (visible or about to disappear),
- `S` = total number of distinct snippets (constant).

Then:

- `P(t) + A(t) = S` for all `t`,
because snippets are never created or destroyed, only moved between “in pool” and “active” states.

The process is:

1. Initially, `A(0) = initial_active_count` subjects are spawned with lifetimes `T_life_i ~ Exp(1 / theta)`.  
   The remaining `S - A(0)` snippets go into `P(0)`.

2. Whenever an active subject’s lifetime expires, `A(t)` decreases by 1, and `P(t)` increases by 1.

3. Each time an inter-arrival clock `Ak ~ Exp(1 / alpha)` fires:
   - If the pool is non-empty, one snippet is drawn,  
     `P(t) -> P(t) - 1`, `A(t) -> A(t) + 1`,
   - A new subject is spawned with its own lifetime `T_life_i`.

This creates a **birth–death style process** for the number of active subjects, driven by two independent exponential mechanisms:

- **“Deaths”** (subjects disappearing) with rate related to `1 / theta`,
- **“Births”** (subjects appearing) with rate `1 / alpha`, modulated by pool availability.

### 2.4. Implementation link

Initialization in `initialize_dynamic_pool`:

```330:349:environment.py
    def initialize_dynamic_pool(self, snippet_pool: list):
        ...
        # Schedule first appearance using exponential distribution
        self._schedule_next_appearance()
        
        # Assign lifetimes to all currently active subjects
        current_time = self._elapsed_sim_seconds()
        for agent in self._agents:
            if getattr(agent, "role", None) == "SUBJECT" and getattr(agent, "visible", True):
                lifetime = random.expovariate(1.0 / self.lifetime_mean_time)
                self.subject_lifetimes[agent] = current_time + lifetime
```

Scheduling new appearances:

```351:357:environment.py
    def _schedule_next_appearance(self):
        """Schedule when the next snippet from the pool will appear."""
        if self.snippet_pool:
            delay = random.expovariate(1.0 / self.appearance_mean_time)
            self.next_appearance_time = self._elapsed_sim_seconds() + delay
        else:
            self.next_appearance_time = None
```

Dynamic updates during the simulation loop:

```394:430:environment.py
    def dynamic_pool_update(self):
        ...
        current_time = self._elapsed_sim_seconds()
        ...
        # Check for subjects whose lifetime has expired (disappearances)
        for subject, expiry_time in list(self.subject_lifetimes.items()):
            if current_time >= expiry_time and getattr(subject, "visible", True):
                expired_subjects.append(subject)
        ...
        for subject in expired_subjects:
            subject.set_visible(False)
            # Return snippet to the pool
            snippet = getattr(subject, "info", "")
            if snippet:
                self.snippet_pool.append(snippet)
            del self.subject_lifetimes[subject]
        ...
        # Check for new appearances
        if self.next_appearance_time is not None and current_time >= self.next_appearance_time:
            new_subject = self._spawn_subject_from_pool()
            ...
            # Schedule next appearance
            self._schedule_next_appearance()
```

### 2.5. Intuitive examples

#### Example C – Moderate churn

Suppose:

- Total snippets `S = 20`,
- `initial_active_count = 5` ⇒ `A(0) = 5`, `P(0) = 15`,
- `lifetime_mean_time = 10.0` seconds ⇒ `theta = 10`, `lambda_life = 0.1`,
- `appearance_mean_time = 5.0` seconds ⇒ `alpha = 5`, `lambda_arr = 0.2`.

Interpretation:

- **Subjects live relatively long** (on average 10 seconds),
- **New subjects appear relatively frequently** (on average every 5 seconds).

You will typically see:

- A **non-zero number of active subjects** most of the time,
- Subjects **coming and going**:
  - Some disappear after a short time,
  - Some stay visible for quite a while,
- Over time, different snippets from the pool are rotated in and out.

Intuitively:

- The system is like a room where **posters** (snippets) are put up on walls (subjects) and later taken down:
  - How long a poster stays up ~ exponential with mean 10 seconds,
  - How often a new poster is put up somewhere in the room ~ exponential with mean 5 seconds,
  - Posters taken down are put back into a storage room (the pool) and may be reused later.

#### Example D – Fast churn

If you set:

- `lifetime_mean_time = 3.0` (short lifetimes),
- `appearance_mean_time = 3.0` (similar arrival rate),

then:

- Individual subjects don’t survive long,
- New subjects appear quickly,
- The set of visible snippets churns rapidly: what knowledge agents see at any moment is a **fast-moving window** over the underlying pool of information.

---

## 3. Summary: decay vs dynamic pool

- **`decay` mode**
  - Each subject gets an independent **exponential lifetime**.
  - Once it disappears, it **never comes back**.
  - Expected number of visible subjects decays as:
    - `E[N(t)] = N0 * exp(-(t - t0) / theta)`.
  - Models **information loss** over time without replacement.

- **`dynamic_pool` mode**
  - Active subjects have exponential lifetimes, **and** new subjects appear with exponential inter-arrival times.
  - Snippets are conserved: they move between a **pool** and **active subjects**.
  - Models a **churning environment** where information sites appear and disappear, but the total underlying information set is fixed.

Together, these two modes let you explore:

- Pure decay of information availability (`decay`), and
- Dynamic turnover of information sources (`dynamic_pool`),

both grounded in a rigorous **continuous-time exponential / Poisson-process** view of event timing.

