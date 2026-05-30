# AgriDefend — Raspberry Pi 5 Deployment Guide

End-to-end, copy-pasteable setup for the team's **exact** existing hardware:
Raspberry Pi 5 (8 GB) · Arducam · BME680 · ADS1115 + capacitive soil moisture ·
MAX485 + RS-485 7-in-1 soil sensor · MH-Z19C CO₂ sensor.

> Train the models on a more powerful machine, copy the three model files to the
> Pi (Section 5), then run in hardware mode (Section 10).

---

## 1. OS Setup

Use **Raspberry Pi OS Bookworm, 64-bit** (required for the Pi 5 — do **not** use
the 32-bit image, TensorFlow will not work on it).

1. Install **Raspberry Pi Imager** on your laptop.
2. Choose Device: *Raspberry Pi 5*, OS: *Raspberry Pi OS (64-bit)*, and your SD card.
3. In the gear/⚙ settings, set hostname, enable SSH, and configure Wi-Fi/locale.
4. Write the image, insert the card, and boot the Pi.

---

## 2. Enable Interfaces

Use `raspi-config` in non-interactive mode:

```bash
sudo raspi-config nonint do_i2c 0        # enable I2C
sudo raspi-config nonint do_spi 0        # enable SPI
sudo raspi-config nonint do_serial_hw 1  # enable serial hardware (UART)
sudo raspi-config nonint do_serial_cons 0  # DISABLE serial login console
```

> **CRITICAL:** `do_serial_cons 0` disables the serial *console* to free
> `/dev/serial0` for the MH-Z19C CO₂ sensor. Without this step the CO₂ sensor
> will not respond.

Enable the camera (libcamera stack, not legacy). Append to
`/boot/firmware/config.txt`:

```ini
camera_auto_detect=1
dtoverlay=vc4-kms-v3d
```

Then reboot:

```bash
sudo reboot
```

---

## 3. Wiring Pinout (Raspberry Pi 5 — GPIO layout differs from Pi 4)

### BME680 climate sensor (I2C)
| Signal | Pi pin | Notes |
|--------|--------|-------|
| VCC | Pin 1 (3.3V) | |
| GND | Pin 6 (GND) | |
| SDA | Pin 3 (GPIO2, I2C Data) | |
| SCL | Pin 5 (GPIO3, I2C Clock) | |

Appears as **0x77** on `i2cdetect`.

### ADS1115 ADC (capacitive soil moisture)
| Signal | Pi pin | Notes |
|--------|--------|-------|
| VCC | Pin 1 (3.3V) | |
| GND | Pin 9 (GND) | |
| SDA | Pin 3 (GPIO2) | shared I2C bus with BME680 |
| SCL | Pin 5 (GPIO3) | shared I2C bus |
| ADDR | Pin 9 (GND) | sets I2C address to **0x48** |
| A0 | — | capacitive soil-moisture sensor output |

### MAX485 TTL↔RS-485 converter (RS-485 7-in-1 soil sensor)
| Signal | Pi pin | Notes |
|--------|--------|-------|
| VCC | Pin 2 (5V) | |
| GND | Pin 14 (GND) | |
| RO | Pin 10 (GPIO15, UART0 RX) | |
| DI | Pin 8 (GPIO14, UART0 TX) | |
| DE | Pin 12 (GPIO18) | tied together with RE |
| RE | Pin 12 (GPIO18) | tied to DE |
| A / B | — | RS-485 A and B terminals of the 7-in-1 soil sensor |

### MH-Z19C CO₂ sensor
| Signal | Pi pin | Notes |
|--------|--------|-------|
| VCC | Pin 4 (5V) | **must be 5V, not 3.3V** |
| GND | Pin 34 (GND) | |
| TX | Pin 10 (GPIO15, UART0 RX) | |
| RX | Pin 8 (GPIO14, UART0 TX) | |

> **UART sharing decision.** The MAX485 and the MH-Z19C both want UART0. Pick one:
>
> **Option A — UART0 for RS-485, second UART for CO₂ (recommended).** The Pi 5 has
> extra UARTs. Enable UART2 by adding to `/boot/firmware/config.txt`:
> ```ini
> dtoverlay=uart2
> ```
> Then wire the MH-Z19C to UART2: **TX → GPIO4 (Pin 7)**, **RX → GPIO5 (Pin 29)**,
> and set the CO₂ port in `src/sensors/co2_sensor.py` from `/dev/serial0` to
> `/dev/ttyAMA2`.
>
> **Option B — UART0 for CO₂ only**, and bit-bang the RS-485 line. Use this only
> if you cannot free a second UART.

### Arducam
Connect the CSI ribbon cable to the **CSI-2** port (labeled **CAM/DISP 1**) — the
Pi 5 has two CSI ports.

---

## 4. Software Installation (run in order)

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv i2c-tools git
git clone https://github.com/nikhilvincentv/crop-stress-detection.git
cd crop-stress-detection
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the I2C devices are detected:

```bash
sudo i2cdetect -y 1
# expect 0x48 (ADS1115) and 0x77 (BME680)
```

---

## 5. Copy Models to the Device

Train on a more powerful machine, then copy the three model files to the Pi:

```bash
scp models/agridefend_multispecies.tflite pi@<PI_IP>:~/crop-stress-detection/models/
scp models/rf_multispecies.pkl            pi@<PI_IP>:~/crop-stress-detection/models/
scp models/metalearner_multispecies.pkl   pi@<PI_IP>:~/crop-stress-detection/models/
```

---

## 6. Configure Crop Species

Edit `src/config.yaml` and set the deployed crop:

```yaml
species: "rice"   # change to your deployed crop
```

Available species slugs (must match the training data exactly):

```
apple        banana       blackgram    chickpea     coconut
coffee       cotton       grapes       jute         kidneybeans
lentil       maize        mango        mothbeans    mungbean
muskmelon    orange       papaya       pigeonpeas   pomegranate
rice         watermelon
```

At deployment the species is set once by the farmer in the mobile app.

---

## 7. Set Up the systemd Service

Update `deploy/agrisafe.service` with the correct paths:

```ini
[Unit]
Description=AgriDefend Edge AI Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/crop-stress-detection
ExecStart=/home/pi/crop-stress-detection/venv/bin/python main.py --mode hardware
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Install and enable it:

```bash
sudo cp deploy/agrisafe.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agrisafe.service
sudo systemctl start agrisafe.service
```

---

## 8. Set Up the Wi-Fi Hotspot

```bash
sudo apt-get install -y hostapd dnsmasq
```

`/etc/hostapd/hostapd.conf`:

```ini
interface=wlan0
ssid=AgriDefend
hw_mode=g
channel=7
wpa=2
wpa_passphrase=agridefend2025
wpa_key_mgmt=WPA-PSK
```

`/etc/dnsmasq.conf`:

```ini
interface=wlan0
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
```

Enable the services:

```bash
sudo systemctl unmask hostapd
sudo systemctl enable hostapd
sudo systemctl start hostapd
sudo systemctl restart dnsmasq
```

---

## 9. Sensor Verification Tests (run in order)

```bash
python test_co2_baud.py
```
* Expected: `CO2: XXX ppm`
* If it fails: confirm the serial console is disabled (Section 2), check 5V power,
  and check TX/RX are not swapped.

```bash
python calibrate_soil.py
```
* Follow the prompts: dry calibration, then wet calibration.
* Save the reported `dry_value` and `wet_value` into `src/config.yaml`.

```bash
python test_all_sensors.py
```
* Expected: all four sensors show green checkmarks.
* If BME680 fails: try `i2c_address: 0x76` in config.
* If CO₂ fails: re-run `test_co2_baud.py` and try swapping the TX/RX wires.

```bash
sudo i2cdetect -y 1
```
* Expected: shows **0x48** and **0x77**.

---

## 10. Full Hardware Mode

```bash
python main.py --mode hardware
```

* First boot takes ~30 s to load the TFLite model.
* Connect your phone to the **AgriDefend** Wi-Fi network.
* Open the React Native app.
* Select your crop species.
* Readings begin every 5 minutes.
