import { createSlice, PayloadAction } from '@reduxjs/toolkit'
import { Paper } from '../../components/ui/PaperCard'

export interface PaperDetail extends Paper {
  url: string
  interpretation?: string
  relatedPapers?: Paper[]
}

export interface PaperFeedback {
  paperId: string
  isPositive: boolean
}

export interface PaperState {
  recommendations: Paper[]
  currentPaper: PaperDetail | null
  loading: boolean
  error: string | null
  hasMore: boolean
  page: number
  interpretationLoading: boolean
  interpretationError: string | null
  feedbacks: Record<string, boolean> // paperId -> isPositive
}

const initialState: PaperState = {
  recommendations: [],
  currentPaper: null,
  loading: false,
  error: null,
  hasMore: true,
  page: 1,
  interpretationLoading: false,
  interpretationError: null,
  feedbacks: {}
}

const paperSlice = createSlice({
  name: 'paper',
  initialState,
  reducers: {
    fetchRecommendationsStart: (state) => {
      state.loading = true
      state.error = null
    },
    fetchRecommendationsSuccess: (state, action: PayloadAction<{ papers: Paper[], hasMore: boolean }>) => {
      state.recommendations = [...state.recommendations, ...action.payload.papers]
      state.hasMore = action.payload.hasMore
      state.page += 1
      state.loading = false
      state.error = null
    },
    fetchRecommendationsFailure: (state, action: PayloadAction<string>) => {
      state.loading = false
      state.error = action.payload
    },
    fetchPaperDetailStart: (state) => {
      state.loading = true
      state.error = null
    },
    fetchPaperDetailSuccess: (state, action: PayloadAction<PaperDetail>) => {
      state.currentPaper = action.payload
      state.loading = false
      state.error = null
    },
    fetchPaperDetailFailure: (state, action: PayloadAction<string>) => {
      state.loading = false
      state.error = action.payload
    },
    clearRecommendations: (state) => {
      state.recommendations = []
      state.page = 1
      state.hasMore = true
    },
    fetchInterpretationStart: (state) => {
      state.interpretationLoading = true
      state.interpretationError = null
    },
    fetchInterpretationSuccess: (state, action: PayloadAction<string>) => {
      if (state.currentPaper) {
        state.currentPaper.interpretation = action.payload
      }
      state.interpretationLoading = false
      state.interpretationError = null
    },
    fetchInterpretationFailure: (state, action: PayloadAction<string>) => {
      state.interpretationLoading = false
      state.interpretationError = action.payload
    },
    clearCurrentPaper: (state) => {
      state.currentPaper = null
      state.interpretationLoading = false
      state.interpretationError = null
    },
    setPaperFeedback: (state, action: PayloadAction<PaperFeedback>) => {
      const { paperId, isPositive } = action.payload
      state.feedbacks[paperId] = isPositive
    },
    clearPaperFeedback: (state, action: PayloadAction<string>) => {
      delete state.feedbacks[action.payload]
    }
  }
})

export const {
  fetchRecommendationsStart,
  fetchRecommendationsSuccess,
  fetchRecommendationsFailure,
  fetchPaperDetailStart,
  fetchPaperDetailSuccess,
  fetchPaperDetailFailure,
  clearRecommendations,
  fetchInterpretationStart,
  fetchInterpretationSuccess,
  fetchInterpretationFailure,
  clearCurrentPaper,
  setPaperFeedback,
  clearPaperFeedback
} = paperSlice.actions

export default paperSlice.reducer 