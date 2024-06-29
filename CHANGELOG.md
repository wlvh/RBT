# Change Log

## [0.2.0] - 2024-06-28

### Added
- `initialize_numpy_arrays` function:
  - Added initialization of `numpy_arrays` dictionary.
  - Converts `data_all` DataFrame columns to NumPy arrays and stores them in `numpy_arrays`.
  - Initializes additional tracking variables such as `cash`, `position`, `net_value`, etc.
  - Checks and handles the presence of `start_date` and `end_date` in `numpy_arrays['datetime']`.

### Updated
- `MAV_Strategy` function:
  - Added the logic for initializing `numpy_arrays` if not provided.
  - Improved data loading and initialization through `initialize_numpy_arrays`.
  - Added calculation and execution of trading signals using moving average strategy.
  - Ensured the function handles date indices correctly.

- `MACD_Strategy` function:
  - Similar updates as `MAV_Strategy` for initialization and trading logic using MACD indicators.

- `RSIV_Strategy` function:
  - Implemented initialization of `numpy_arrays` using `initialize_numpy_arrays`.
  - Added logic for trading signals based on RSI and moving average indicators.
  - Enhanced error handling for date indices.

- `WRSIStrategy` function:
  - Updated to include initialization using `initialize_numpy_arrays`.
  - Added logic for WRSI-based trading signals.
  - Improved handling of date indices and trading execution.

- `calculate_RSIsignals` function:
  - Refined signal calculation for RSI-based strategies.
  - Added logic to handle live trading scenarios.

- `calculate_MAVsignals` function:
  - Added logic to calculate trading signals based on moving average crossovers.
  - Enhanced handling of buy/sell conditions and signals.

- `calculate_MACDsignals` function:
  - Improved signal calculation logic for MACD-based strategies.
  - Updated handling of live trading scenarios and added detailed printing of conditions.

- `calculate_VWAPsignals` function:
  - Added logic to calculate VWAP-based trading signals.
  - Enhanced handling of buy/sell conditions and signals.

### Fixed
- Bug in date matching and handling within `initialize_numpy_arrays` and strategy functions.
- Corrected initialization of `initial_date` and `end_idx` indices.
- Fixed signal calculation logic to ensure correct handling of rolling windows and shifting arrays.

### Removed
- Deprecated or redundant code snippets that were not contributing to the updated logic.

## [0.1.0] - 2023-12-28
### First Version
