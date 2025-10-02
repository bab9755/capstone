from agents import knowledgeAgent, Villager
from vi import Config, Simulation, Window
from subjects import SubjectAgent
from environment import Environment
from constants import WIDTH, HEIGHT
from story_registry import create_story_environment, story_registry

def run_eldoria_mystery():
    """Run the Eldoria Village Mystery simulation"""
    print("🌍 Welcome to the Eldoria Village Mystery!")
    print("🔍 Three investigation robots are about to explore the village...")
    print("📖 They must piece together the story of the Lost Artifact!")
    print("=" * 60)
    
    # Create the story environment with sites and registry (kept for summary/stats)
    story_sites = create_story_environment(WIDTH, HEIGHT, seed=3)
    
    # Create simulation
    config = Config(window=Window(WIDTH, HEIGHT), seed=3)
    simulation = Environment(config)
    
    # Spawn 3 knowledge agents (robots) to investigate
    simulation.batch_spawn_agents(2, knowledgeAgent, images=["images/robot.png"])

    # Spawn 5 subject agents with a small split story
    fragments = [
        "A strange signal echoed at dawn...",
        "Villagers saw lights above the mill...",
        "Scorch marks formed a perfect ring...",
        "A fragment hummed near the well...",
        "Coordinates etched on stone pointed north."
    ]
    # Spawn one-by-one and assign fragment to each newly spawned subject
    for fragment in fragments:
        simulation.batch_spawn_agents(1, SubjectAgent, images=["images/villager.png"])  # returns Simulation (chainable)
        # Grab the latest subject agent and assign its info
        subjects = [a for a in simulation._agents if getattr(a, "role", None) == "SUBJECT"]
        if subjects:
            subjects[-1].info = fragment

    
    print(f"✅ Environment created with {len(story_sites)} story sites")
    print(f"🤖 {2} investigation robots deployed")
    print("🚀 Starting simulation...")
    print("=" * 60)
    
    return simulation

def print_final_summary():
    """Print a final summary of the investigation"""
    print("\n" + "=" * 60)
    print("🏁 INVESTIGATION COMPLETE - FINAL REPORT")
    print("=" * 60)
    
    progress = story_registry.get_discovery_progress()
    
    print(f"📊 Discovery Statistics:")
    print(f"   • Total discoveries: {progress['total_discoveries']}")
    print(f"   • Unique stories found: {progress['unique_stories_found']}/{progress['total_stories']}")
    print(f"   • Completion rate: {progress['completion_percentage']:.1f}%")
    
    print(f"\n🤖 Agent Performance:")
    for agent_id, discoveries in progress['agent_discoveries'].items():
        print(f"   • Agent {agent_id}: {discoveries} discoveries")
    
    if progress['unique_stories_found'] == progress['total_stories']:
        print(f"\n🎉 MYSTERY SOLVED! All story fragments have been discovered!")
        print(f"📖 The robots have successfully pieced together the Eldoria mystery!")
    else:
        print(f"\n🔍 Investigation incomplete. {progress['total_stories'] - progress['unique_stories_found']} story fragments remain undiscovered.")
    
    print("=" * 60)

# Create and run the story environment
if __name__ == "__main__":
    try:
        simulation = run_eldoria_mystery()
        simulation.run()
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
        simulation.print_agents_context()
        simulation.stop()
    finally:
        print_final_summary()
