WIDTH=750
HEIGHT=750

# system_prompt = """
# You are a summarization model focused on accuracy, fidelity, and brevity.

# Summarize only the content of the input text.

# Do not include phrases like “Here’s a summary” or any commentary.

# If the payload appears empty, irrelevant, or non-informative, return an empty string.

# Preserve the original meaning as closely as possible; avoid speculation or creativity.

# The response should be a single plain-text summary paragraph.

# Avoid long sentences and keep each sentence short, to the point, as long at it captures the main idea of the input text.
# The summary must be 150 words MAXIMUM.

# """

system_prompt = """
you are a data fusion module for a collaborative robot swarm. Your function is to synthesize two partial observations into a single, more complete, and factually dense description of the environment.
You will be given a prompt that is either a combination of a private information of the agent with its current summary OR a combination of its received summaries and observations from other agents and its current summary.

      YOUR TASK: Perform a factual merge of the two observations to create a single, unified log.
      Follow the following steps:

      Identify & De-duplicate: Identify all unique objects, attributes, and spatial relationships from both sources. If both sources describe the same object (e.g., "a building"), treat them as one.

      Reconcile & Enhance: If descriptions for the same object differ, merge the details. Use the most specific and complete information from both.

      Lexical and Semantic Consistency: Ensure that the summary is consistent with the lexical and semantic information in the prompt.

      Example 1 (Detail): "a tree" + "a tall oak tree" = "a tall oak tree". This is just an example, don't add this information to the summary

      Example 2 (Attributes): "a blue truck" + "a truck with a dented fender" = "a blue truck with a dented fender". This is just an example, don't add this information to the summary

      Integrate & Spatialize: Combine the complete, de-duplicated set of facts into a single, spatially coherent paragraph. The goal is to describe a larger, combined area.

      Example (Spatial): "a rock is north of the stream" + "a fence is south of the stream" = "a rock and a fence are on opposite sides of the stream". This is just an example, don't add this information to the summary

      OUTPUT RULES:

      Format: YOUR ENTIRE RESPONSE MUST BE ONLY THE FINAL SYNTHESIZED SUMMARY.

      Style: Single paragraph of plain text. No markdown, lists, or headings.

      Objectivity: Be literal, factual, and geometric.

      Length: Must not exceed 150 words.

"""

fragments = [
  "Families arrive at the city zoo on a sunny Saturday morning.",
  "Children run toward the lion enclosure, excited for feeding time.",
  "The zookeeper throws meat to the lions as people watch.",
  "A girl feeds ducks at the pond even though a sign says not to.",
  "A vendor sells ice cream beside a bench near the bird cages.",
  "An elderly couple sits on the bench, watching colorful parrots mimic people.",
  "Inside the reptile house, students sketch snakes for a biology project.",
  "The reptile area feels warm and humid compared to outside.",
  "A loudspeaker announces the penguin show starting in ten minutes.",
  "Crowds start walking toward the penguin stadium.",
  "A child loses his balloon, and it floats up into a tall tree.",
  "The parents laugh as the balloon drifts away.",
  "The smell of popcorn spreads through the zoo.",
  "Animal sounds mix with people’s laughter in the background."
]



ground_truth_text = """
On a sunny Saturday morning, families arrive at the city zoo, filling it with excitement and chatter. Children race toward the lion enclosure just in time for feeding, where the zookeeper tosses meat to the roaring lions as the crowd watches in awe. Nearby, a little girl sneaks pieces of bread to ducks at the pond despite a sign warning not to feed them. A vendor sells ice cream beside a bench near the bird cages, where an elderly couple sits watching colorful parrots mimic people’s voices. Inside the reptile house, a group of students sketches snakes for their biology project, surrounded by the warm, humid air. Over the loudspeaker, an announcement echoes that the penguin show will start in ten minutes, prompting crowds to head toward the stadium. A child loses his balloon, and it floats high into a tall tree while his parents laugh as it drifts away. The scent of popcorn drifts through the zoo, mingling with the animal sounds and the laughter of visitors in the vibrant atmosphere.

"""



ground_truth = [
    "The university hosts an annual career fair in a main hall crowded with company booths as hundreds of students walk through.",
    "The atmosphere is busy, energetic, tense, and exciting throughout the event.",
    "Recruiters from tech companies, consulting firms, and startups hand out brochures and discuss internships and full-time jobs with students.",
    "Students hold their resumes tightly, rehearse elevator pitches, and collect branded tote bags as they navigate the fair.",
    "Business cards change hands constantly as students network and exchange contact information throughout the event.",
    "Panels occur in nearby rooms while workshops teach career and interview skills to help students prepare.",
    "Alumni give advice to current students on making a good impression and navigating the career fair experience.",
    "Career counselors guide students individually and match them with fitting companies based on their interests and qualifications.",
    "Small groups of students talk outside the main area and share which recruiters seemed most interested in their profiles.",
    "The day feels long and exhausting, but the event is filled with new opportunities where quick conversations can lead to future jobs."
]


ground_truth_summary = """

The university career fair is a lively event where students meet recruiters from various industries to explore internships and job opportunities. The atmosphere is charged with ambition as attendees practice pitches, exchange resumes, and network across dozens of company booths. Panels and alumni sessions provide guidance on career development and interview preparation, while counselors assist students in identifying relevant employers. Between the crowded aisles and buzzing conversations, the fair represents a crucial moment of connection — an opportunity for students to transform brief introductions into meaningful professional paths.

"""

ground_truth_facts = [
    "The university hosts an annual career fair filled with company booths and hundreds of students.",
    "Recruiters from tech companies, consulting firms, and startups offer brochures and discuss job opportunities.",
    "Students network actively by exchanging resumes, elevator pitches, and business cards.",
    "The event creates career opportunities as short conversations can lead to future jobs."
]
