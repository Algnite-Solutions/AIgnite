import { configureStore } from '@reduxjs/toolkit'
import userReducer from './slices/userSlice'
import paperReducer from './slices/paperSlice'
import favoritesReducer from './slices/favoritesSlice'

export const store = configureStore({
  reducer: {
    user: userReducer,
    paper: paperReducer,
    favorites: favoritesReducer
  }
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch 