# Contributing to Crop Stress Early Warning System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/crop-stress-early-warning.git
   cd crop-stress-early-warning
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

## Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

## Testing

- Write unit tests for new features
- Run tests before submitting PR:
  ```bash
  pytest tests/
  ```
- Ensure simulation mode works:
  ```bash
  python main.py --mode simulation
  ```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for new functionality
4. Update documentation (README.md, docstrings)
5. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a Pull Request with:
   - Clear description of changes
   - Reference to related issues
   - Test results

## Areas for Contribution

### High Priority
- Additional sensor integrations (soil moisture, light sensors)
- Mobile app for real-time monitoring
- Cloud data synchronization
- Advanced ML models (LSTM, attention mechanisms)
- Multi-crop calibration profiles

### Medium Priority
- Web dashboard for visualization
- Automated alerts/notifications
- Integration with weather APIs
- Power optimization for solar deployment
- Multi-language support

### Documentation
- Tutorial videos
- Hardware assembly guide with photos
- Troubleshooting guide expansion
- Translation to other languages

## Bug Reports

When reporting bugs, please include:
- System information (OS, Python version)
- Hardware configuration (if applicable)
- Steps to reproduce
- Expected vs actual behavior
- Error messages and logs
- Screenshots if relevant

## Feature Requests

For feature requests, please:
- Check existing issues first
- Describe the use case
- Explain why it would be valuable
- Suggest implementation approach (optional)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

Feel free to open an issue for questions or join discussions!
