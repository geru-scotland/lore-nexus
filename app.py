"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: app.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import re
import sys
from enum import Enum
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.flair.model import LoreNexusFlairModel

class UserOptions(Enum):
    UNVEIL_LORE = 1
    EXIT = 2

    def __str__(self):
        if self == UserOptions.UNVEIL_LORE:
            return "Unveil hidden lore from a name 🌌 (Reveal the secrets)"
        elif self == UserOptions.EXIT:
            return "Exit the Lore Nexus"


class LoreNexusApp:
    def __init__(self):
        """
        """
        config = self.load_config()
        model_path = config['model']['path']
        self.lore_nexus = LoreNexusFlairModel(mode="cli_app", model_path=model_path)
        self.display_title()
        print("📜 The Lore Nexus is ready to unveil the mysteries of any name you provide 📜 ")

    def load_config(self):
        """
        """
        config_path = "config.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def display_title(self):
        """
        Thanks to https://fsymbols.com/generators/carty/
        """
        title = """
██╗░░░░░░█████╗░██████╗░███████╗ ███╗░░██╗███████╗██╗░░██╗██╗░░░██╗░██████╗
██║░░░░░██╔══██╗██╔══██╗██╔════╝ ████╗░██║██╔════╝╚██╗██╔╝██║░░░██║██╔════╝
██║░░░░░██║░░██║██████╔╝█████╗░░ ██╔██╗██║█████╗░░░╚███╔╝░██║░░░██║╚█████╗░
██║░░░░░██║░░██║██╔══██╗██╔══╝░░ ██║╚████║██╔══╝░░░██╔██╗░██║░░░██║░╚═══██╗
███████╗╚█████╔╝██║░░██║███████╗ ██║░╚███║███████╗██╔╝╚██╗╚██████╔╝██████╔╝
╚══════╝░╚════╝░╚═╝░░╚═╝╚══════╝ ╚═╝░░╚══╝╚══════╝╚═╝░░╚═╝░╚═════╝░╚═════╝░                                                                           
        """
        print(title)

    def unveil_lore_from_name(self, name):
        """
        """
        return self.lore_nexus.predict_name(name)

    def display_results(self, name, results):
        """
        """
        print("\n" + "╔" + "═" * 52 + "╗")
        title_text = f"✦ Ancient Lore Archive: Secrets of '{name}' ✦"
        print("║{:^52}║".format(title_text))
        print("╚" + "═" * 52 + "╝")

        for i, (prediction, (label, confidence)) in enumerate(results.items()):
            if i == 0:
                print(f"⟡ {prediction}: {label} with confidence {confidence:.4f} 📜 ")
            else:
                print(f"{prediction}: {label} with confidence {confidence:.4f}")

    def start(self):
        """
        """
        while True:
            print("\nChoose an option:")
            print("1 - Unveil hidden lore from a name 🌌 (Reveal the story and secrets behind)")
            print("2 - Exit the Lore Nexus")

            try:

                choice = input("Your choice: ").strip()

                if choice == str(UserOptions.UNVEIL_LORE.value):
                    name = input("Enter a name to uncover its Lore: ").strip()

                    if not name or not re.match("^[a-zA-Z0-9 ]*$", name):
                        print("Please provide a correct name.")
                        continue

                    result = self.unveil_lore_from_name(name)
                    self.display_results(name, result)

                elif choice == str(UserOptions.EXIT.value):
                    print("Closing the Lore Nexus. Farewell, traveler!")
                    break
                else:
                    print("Invalid choice, please try again.")
            except Exception as e:
                print(f"An error occurred: {e}")
                break

if __name__ == "__main__":
    app = LoreNexusApp()
    app.start()
