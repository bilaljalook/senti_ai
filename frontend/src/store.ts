import { configureStore } from '@reduxjs/toolkit';
import bitcoinReducer from './features/bitcoinSlice';

export const store = configureStore({
  reducer: {
    bitcoin: bitcoinReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
