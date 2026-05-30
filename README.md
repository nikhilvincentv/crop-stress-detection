Crop Stress Early Warning System

Multi-Modal Predictive System for Early Detection of Crop Stress

How to Run




Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt 
   ```
## Hardware Setup (Raspberry Pi 4)

To run this system in **Hardware Mode**, assemble the following components (~$180):
1. **MH-Z19B CO2 Sensor** (connected via Serial/UART)
2. **BME680 Sensor** (connected via I2C)
3. **Pi NoIR Camera** (connected via CSI port)
4. **Soil Moisture Sensor** (connected via ADC/SPI)

### Software Installation on Pi
1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd crop-stress-detection
   ```
2. Set up environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Enable I2C, SPI, and Serial in `raspi-config`:
   ```bash
   sudo raspi-config
   ```

## Running the System

### Simulation Mode (No Hardware)
```bash
python main.py --mode simulation
```

<img width="1080" height="1350" alt="Untitled (Instagram Post (45))" src="https://github.com/user-attachments/assets/599778a5-889b-417d-bcfd-ed9c3b00699a" />
<img width="940" height="788" alt="Untitled (Facebook Post)" src="https://github.com/user-attachments/assets/df5e1d37-7e01-4c24-b90d-1a69cf241971" />

### PyTorch LSTM Training
```bash
python train_lstm.py
```

### Real Hardware Mode
```bash
python main.py --mode hardware
```

For a full, copy-pasteable Raspberry Pi 5 setup (wiring, interfaces, systemd, Wi-Fi hotspot, sensor tests), see [`RASPBERRY_PI_DEPLOYMENT.md`](RASPBERRY_PI_DEPLOYMENT.md).

![Hardware Layout](https://github.com/user-attachments/assets/dda0d567-755d-4496-91f0-6ccaaed2ee29)

