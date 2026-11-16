# AgriDefend

AgriDefend is a small solar-powered device that watches a crop and warns a farmer about stress before the plants show symptoms. A camera looks at the leaves. A soil probe and a few other sensors read what's happening at the roots and in the air. A Raspberry Pi 5 combines both and gives a stress reading. It all runs on the device, so it needs no cloud and no cell signal.

I started this with my co-founder George Xu, and I lead the engineering.

## Results

- 87% accuracy in real-world testing, across five crop-health classes
- 1.4 seconds per prediction on the Pi
- $205 in parts
- Running on two farms in Washington: 21 Acres Farm and Hawthorn Farms
- 1st place at the 2026 Washington State Science and Engineering Fair
- Bronze medal at the Genius Olympiad, AI category
- Provisional patent application filed May 2026

## Links

- Paper: [AgriDefend: A $205 Offline Edge Device that Detects Crop Stress Before Visible Symptoms via Multimodal AI Fusion](https://arxiv.org/abs/2610.34861), with George Xu and Elizabeth Forbes
- Project page: https://nikhilvincent.dev/projects/agrisafe-edge-ml
- Website: https://agridefend.org
- What broke when we put it in a field: https://nikhilvincent.dev/blog/agridefend-field-testing
- Getting the Pi to read a soil probe: https://nikhilvincent.dev/blog/raspberry-pi-modbus-bridge

## Pictures

<img width="1080" height="1350" alt="AgriDefend project graphic" src="https://github.com/user-attachments/assets/599778a5-889b-417d-bcfd-ed9c3b00699a" />
<img width="940" height="788" alt="AgriDefend project graphic" src="https://github.com/user-attachments/assets/df5e1d37-7e01-4c24-b90d-1a69cf241971" />

![Hardware Layout](https://github.com/user-attachments/assets/dda0d567-755d-4496-91f0-6ccaaed2ee29)

## How it works

1. **Leaves.** The camera takes a picture of the plant. A MobileNetV2 image model trained on the PlantVillage dataset and exported as an INT8 TFLite file scores how stressed the leaves look.
2. **Soil and air.** The sensors read soil moisture, soil chemistry, CO2, temperature, humidity and gas. A random forest turns those readings into a second score.
3. **Fusion.** A small meta-learner combines the two scores into one stress reading.

To retrain everything, `scripts/download_datasets.py` pulls the public datasets (PlantVillage, Kaggle Crop Recommendation, NASA SRDB, COSORE and USDA Cook Agronomy). Then run `train_cnn.py`, `train_rf.py` and `train_fusion.py` in the same folder.

## Hardware

- Raspberry Pi 5
- BME680 for temperature, humidity, pressure and gas (I2C)
- ADS1115 with a capacitive soil moisture sensor
- MAX485 board with an RS-485 7-in-1 soil probe, which reads moisture, temperature, conductivity, pH, nitrogen, phosphorus and potassium
- MH-Z19C CO2 sensor
- Arducam camera

Wiring, interface setup, the systemd service and the Wi-Fi hotspot are all in [RASPBERRY_PI_DEPLOYMENT.md](RASPBERRY_PI_DEPLOYMENT.md).

## Run it

```bash
git clone https://github.com/nikhilvincentv/crop-stress-detection
cd crop-stress-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No hardware needed for simulation mode:

```bash
python main.py --mode simulation
```

On the Pi with the sensors wired up:

```bash
python main.py --mode hardware
```

There's also an LSTM experiment: `python train_lstm.py`.

## What's in here

- `main.py` and `src/` run the system: sensor drivers, inference and the advice generator
- `scripts/` downloads data and trains the models
- `models/` holds the trained model files
- `plots/` has charts from the training runs
- `deploy/` has the systemd service for the Pi

Some older files, like `docs/architecture.md`, still describe the first Raspberry Pi 4 prototype. The current build uses a Pi 5.

Built by Nikhil Vincent and George Xu. More of my work is at https://nikhilvincent.dev.
