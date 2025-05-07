import { createSlice, PayloadAction } from '@reduxjs/toolkit'
import { Paper } from '../../components/ui/PaperCard'

interface FavoritesState {
  papers: Paper[]
  loading: boolean
  error: string | null
}

const initialState: FavoritesState = {
  papers: [],
  loading: false,
  error: null
}

const favoritesSlice = createSlice({
  name: 'favorites',
  initialState,
  reducers: {
    // 添加论文到收藏
    addFavorite: (state, action: PayloadAction<Paper>) => {
      // 检查是否已存在此论文
      const exists = state.papers.some(paper => paper.id === action.payload.id)
      if (!exists) {
        state.papers.push(action.payload)
      }
    },
    
    // 从收藏中移除论文
    removeFavorite: (state, action: PayloadAction<string>) => {
      state.papers = state.papers.filter(paper => paper.id !== action.payload)
    },
    
    // 加载收藏论文开始
    loadFavoritesStart: (state) => {
      state.loading = true
      state.error = null
    },
    
    // 加载收藏论文成功
    loadFavoritesSuccess: (state, action: PayloadAction<Paper[]>) => {
      state.papers = action.payload
      state.loading = false
      state.error = null
    },
    
    // 加载收藏论文失败
    loadFavoritesFailure: (state, action: PayloadAction<string>) => {
      state.loading = false
      state.error = action.payload
    }
  }
})

export const {
  addFavorite,
  removeFavorite,
  loadFavoritesStart,
  loadFavoritesSuccess,
  loadFavoritesFailure
} = favoritesSlice.actions

export default favoritesSlice.reducer 