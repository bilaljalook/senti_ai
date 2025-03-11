import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface BitcoinState {
  historicalData: any[];
  predictedData: any[];
  loading: boolean;
  error: string | null;
}

const initialState: BitcoinState = {
  historicalData: [],
  predictedData: [],
  loading: false,
  error: null,
};

export const bitcoinSlice = createSlice({
  name: 'bitcoin',
  initialState,
  reducers: {
    setHistoricalData: (state, action: PayloadAction<any[]>) => {
      state.historicalData = action.payload;
    },
    setPredictedData: (state, action: PayloadAction<any[]>) => {
      state.predictedData = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const { setHistoricalData, setPredictedData, setLoading, setError } = bitcoinSlice.actions;

export default bitcoinSlice.reducer;
