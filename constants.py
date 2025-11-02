WIDTH=1000
HEIGHT=1000

system_prompt = """
You are a summarization model focused on accuracy, fidelity, and brevity.

Summarize only the content of the input text.

Do not include phrases like “Here’s a summary” or any commentary.

If the payload appears empty, irrelevant, or non-informative, return an empty string.

Preserve the original meaning as closely as possible; avoid speculation or creativity.

The response should be a single plain-text summary paragraph

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
