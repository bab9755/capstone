from typing import Dict, List, Tuple
import random

class StoryRegistry:
    """Global registry for story information associated with sites"""
    
    def __init__(self):
        self.site_info: Dict[Tuple[float, float], str] = {}
        self.story_sites: List[Dict] = []
        self.discovery_log: List[Dict] = []  # Track all discoveries across all agents
        
    def register_site(self, x: float, y: float, info: str, sprite: str):
        """Register a site with its story information"""
        # Use a small tolerance for position matching (sites might not be exactly at the same position)
        self.site_info[(x, y)] = info
        self.story_sites.append({
            "position": (x, y),
            "info": info,
            "sprite": sprite
        })
        
    def get_info_at_position(self, x: float, y: float, tolerance: float = 30.0) -> str:
        """Get story information at a given position with tolerance"""
        for (site_x, site_y), info in self.site_info.items():
            distance = ((x - site_x) ** 2 + (y - site_y) ** 2) ** 0.5
            if distance <= tolerance:
                return info
        return None
        
    def get_all_story_info(self) -> List[str]:
        """Get all story information for debugging/analysis"""
        return list(self.site_info.values())
        
    def get_story_summary(self) -> str:
        """Get a summary of all story fragments"""
        all_info = self.get_all_story_info()
        return "\n".join([f"• {info}" for info in all_info])
    
    def log_discovery(self, agent_id: int, story_info: str, timestamp: int):
        """Log a story discovery for tracking progress"""
        self.discovery_log.append({
            "agent_id": agent_id,
            "story_info": story_info,
            "timestamp": timestamp
        })
    
    def get_discovery_progress(self) -> Dict:
        """Get discovery progress statistics"""
        unique_stories = set()
        agent_discoveries = {}
        
        for discovery in self.discovery_log:
            unique_stories.add(discovery["story_info"])
            agent_id = discovery["agent_id"]
            if agent_id not in agent_discoveries:
                agent_discoveries[agent_id] = 0
            agent_discoveries[agent_id] += 1
        
        return {
            "total_discoveries": len(self.discovery_log),
            "unique_stories_found": len(unique_stories),
            "total_stories": len(self.site_info),
            "completion_percentage": (len(unique_stories) / len(self.site_info)) * 100 if self.site_info else 0,
            "agent_discoveries": agent_discoveries
        }

# Global story registry instance
story_registry = StoryRegistry()

# The Lost Artifact of Eldoria - Story Sites
STORY_SITES = [
    # Village Characters
    {"sprite": "images/fighter3.png", "info": "The village elder, Marcus, was the first to discover the glowing artifact near the old oak tree. He said it pulsed with an otherworldly blue light."},
    {"sprite": "images/fighter3.png", "info": "Sarah, the village historian, claims the artifact matches descriptions in ancient texts about the 'Crystal of Eternal Wisdom' from the lost civilization of Aetheria."},
    {"sprite": "images/fighter3.png", "info": "The village mechanic, Tom, found strange metallic fragments near the artifact site. He believes they're not from this world."},
    
    # Environmental Clues
    {"sprite": "images/obstacle.png", "info": "The old stone monument at the village center has newly carved symbols that appeared the night the artifact vanished. They seem to point toward the northern mountains."},
    {"sprite": "images/triangle.png", "info": "Three triangular markings were found burned into the ground where the artifact was discovered, forming a perfect equilateral triangle."},
    {"sprite": "images/red.png", "info": "Red crystalline dust was scattered around the discovery site. Analysis shows it contains elements not found on Earth."},
    
    # Mysterious Events
    {"sprite": "images/green.png", "info": "Witnesses report seeing a green flash of light in the sky the night the artifact disappeared. Some say it was a spacecraft."},
    {"sprite": "images/white.png", "info": "The village well water turned crystal clear and began glowing faintly after the artifact's disappearance. It now has healing properties."},
    {"sprite": "images/fighter2.png", "info": "Military personnel in unmarked vehicles arrived the day after the artifact vanished. They questioned everyone and left without explanation."},
    {"sprite": "images/fighter3.png", "info": "A second military team returned a week later, this time with strange scanning equipment. They seemed to be looking for something specific."},
    
    # Additional Clues
    {"sprite": "images/robot.png", "info": "The village's old radio began picking up strange signals after the artifact disappeared. The signals contain mathematical patterns that seem to be coordinates."},
    {"sprite": "images/robot.png", "info": "Local wildlife has been acting strangely - birds flying in perfect geometric patterns, deer gathering in circles, and fish swimming in synchronized schools."},
]

def create_story_environment(width: int, height: int, seed: int = 3):
    """Create story sites and register them in the global registry"""
    random.seed(seed)
    
    # Clear any existing story data
    story_registry.site_info.clear()
    story_registry.story_sites.clear()
    
    created_sites = []
    
    for site in STORY_SITES:
        x = random.randint(50, width - 50)  # Keep margin from edges
        y = random.randint(50, height - 50)
        
        # Register the site with story information
        story_registry.register_site(x, y, site["info"], site["sprite"])
        
        created_sites.append({
            "sprite": site["sprite"],
            "x": x,
            "y": y,
            "info": site["info"]
        })
    
    return created_sites
