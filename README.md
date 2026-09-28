# Soil pH Prediction from Images

An attempt to predict soil pH values from smartphone photos of soil samples taken in Gambia.

## What it does

Trains an EfficientNet-B0 image regression model to predict soil pH directly from photos. Also includes a classical baseline using Lab* colour features (Ridge, Random Forest, etc.). Images were matched to ground-truth pH readings from field sensor measurements, then split by farm so the model is tested on farms it has never seen.

## Results

Not successful. On held-out farms the model achieved:

- **MAE: 0.596 pH units**
- **R²: -0.478**

An R² below zero means the model is worse than just predicting the average pH for every sample — it actively misfires. The colour baseline performed similarly poorly.

## Why it didn't work

pH does not have a strong relationship with what a camera captures. Research confirms that soil pH correlates with iron oxides (redness) and organic matter (darkness), but these are subtle signals easily swamped by lighting conditions, photo angle, soil moisture at time of photo, and differences between farms. The model likely learned farm-specific colour quirks during training that don't generalise to new farms, which is exactly what the negative R² reflects.

To make this work you'd probably need hyperspectral or near-infrared imaging rather than standard RGB photos.

## Running it

```bash
uv run python main.py --train          # train model
uv run python main.py --eval           # evaluate on holdout farms 10 & 11
uv run python main.py --predict img.jpeg  # predict pH for a single image
uv run python color_model.py           # run the colour-feature baseline
```

Requires a `downloaded_photos/` directory of images named `f<farm>-<sample>.jpeg` and the ground-truth Excel file (not included in this repo).
