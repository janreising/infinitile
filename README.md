[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# infinitile
Infinitile is a tool for generating infinite RPG maps using a mix of procedural generation and AI-driven image synthesis (via Stable Diffusion). Start from a single tile and expand outward in any direction—forever.

> ⚠️ This project is in early development. Expect dragons and rough edges.

---

## Project Vision

The core idea is to blend **algorithmic map logic** with **AI-generated visuals** to create a tool that:
- Starts with a user-defined or random **central tile**
- Expands **infinitely outward** with logical terrain generation
- Uses **Stable Diffusion (or similar)** to generate matching visuals
- Allows **bi-directional growth** and consistent world appearance
- Supports **exporting maps** (image slices, tile coordinates, maybe game engine formats)

Target use cases:
- Tabletop RPG map prep
- Sandbox game prototyping
- Generative art / infinite world building

---

## TODOs

### Core Features
- [ ] Define base tile structure and metadata format
- [ ] Implement procedural neighbor rule system
- [ ] Set up Stable Diffusion pipeline for visual tile generation
- [ ] Create tile stitching / blending logic
- [ ] Add map explorer UI (basic viewer or notebook)

### Dev Setup
- [ ] Basic CLI interface for testing tile expansion
- [ ] Add support for image caching and seeding
- [ ] Add initial example maps and generation seeds

### Experiments & Stretch Goals
- [ ] Directional generation bias (e.g., coastlines, mountain ranges)
- [ ] Integrate trained LoRA / ControlNet for style-specific maps
- [ ] Web-based interface with pan + zoom
- [ ] Save and resume map state

---

## ☕ Support

If you like this project and want to support continued development:

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://www.buymeacoffee.com/janreising)

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).  
You are free to use, modify, and share the code and generated content for non-commercial purposes, with attribution.  
For commercial use, please contact the author.
