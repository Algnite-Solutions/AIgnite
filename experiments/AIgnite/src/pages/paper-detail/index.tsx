import { View, Text } from '@tarojs/components'
import React, { useEffect } from 'react'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import {
  fetchPaperDetailStart,
  fetchPaperDetailSuccess,
  fetchPaperDetailFailure,
  fetchInterpretationStart,
  fetchInterpretationSuccess,
  fetchInterpretationFailure,
  clearCurrentPaper
} from '../../store/slices/paperSlice'
import CustomButton from '../../components/ui/Button'
import Taro from '@tarojs/taro'
import './index.scss'

const PaperDetail = () => {
  const dispatch = useAppDispatch()
  const {
    currentPaper,
    loading,
    error,
    interpretationLoading,
    interpretationError
  } = useAppSelector((state: any) => state.paper)

  const fetchPaperDetail = async (paperId: string) => {
    try {
      dispatch(fetchPaperDetailStart())
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      const mockPaper = {
        id: paperId,
        title: '深度学习在自然语言处理中的应用',
        authors: ['张三', '李四'],
        abstract: '本文探讨了深度学习技术在自然语言处理领域的最新进展...',
        tags: ['深度学习', 'NLP'],
        url: 'https://arxiv.org/abs/2103.12345',
        relatedPapers: [
          {
            id: '2',
            title: '计算机视觉中的注意力机制研究',
            authors: ['王五', '赵六'],
            abstract: '注意力机制在计算机视觉任务中发挥着越来越重要的作用...',
            tags: ['计算机视觉', '注意力机制']
          }
        ]
      }
      dispatch(fetchPaperDetailSuccess(mockPaper))
    } catch (err) {
      dispatch(fetchPaperDetailFailure(err instanceof Error ? err.message : '获取论文详情失败'))
    }
  }

  const fetchInterpretation = async () => {
    if (!currentPaper) return

    try {
      dispatch(fetchInterpretationStart())
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1500))
      const mockInterpretation = '这篇论文主要研究了深度学习在自然语言处理领域的应用。作者提出了一个新的模型架构，该架构在多个基准测试中都取得了优异的结果。论文的创新点在于...'
      dispatch(fetchInterpretationSuccess(mockInterpretation))
    } catch (err) {
      dispatch(fetchInterpretationFailure(err instanceof Error ? err.message : '获取论文解读失败'))
    }
  }

  const handleViewPaper = () => {
    if (currentPaper?.url) {
      Taro.navigateTo({
        url: `/pages/web-view/index?url=${encodeURIComponent(currentPaper.url)}`
      })
    }
  }

  const handleRelatedPaperClick = (paperId: string) => {
    Taro.navigateTo({
      url: `/pages/paper-detail/index?id=${paperId}`
    })
  }

  useEffect(() => {
    const params = Taro.getCurrentInstance().router?.params
    if (params?.id) {
      fetchPaperDetail(params.id as string)
    }
    return () => {
      dispatch(clearCurrentPaper())
    }
  }, [])

  if (loading) {
    return (
      <View className='paper-detail-page loading'>
        <Text>加载中...</Text>
      </View>
    )
  }

  if (error || !currentPaper) {
    return (
      <View className='paper-detail-page error'>
        <Text>{error || '论文不存在'}</Text>
      </View>
    )
  }

  return (
    <View className='paper-detail-page'>
      <View className='header'>
        <Text className='title'>{currentPaper.title}</Text>
        <Text className='authors'>{currentPaper.authors.join(', ')}</Text>
      </View>

      <View className='content'>
        <View className='section'>
          <Text className='section-title'>摘要</Text>
          <Text className='abstract'>{currentPaper.abstract}</Text>
        </View>

        <View className='section'>
          <Text className='section-title'>标签</Text>
          <View className='tags'>
            {currentPaper.tags.map(tag => (
              <Text key={tag} className='tag'>{tag}</Text>
            ))}
          </View>
        </View>

        {currentPaper.interpretation ? (
          <View className='section'>
            <Text className='section-title'>AI 解读</Text>
            <Text className='interpretation'>{currentPaper.interpretation}</Text>
          </View>
        ) : (
          <View className='section'>
            <CustomButton
              type='primary'
              block
              onClick={fetchInterpretation}
              disabled={interpretationLoading}
            >
              {interpretationLoading ? '生成解读中...' : '生成 AI 解读'}
            </CustomButton>
            {interpretationError && (
              <Text className='error-message'>{interpretationError}</Text>
            )}
          </View>
        )}

        {currentPaper.relatedPapers && currentPaper.relatedPapers.length > 0 && (
          <View className='section'>
            <Text className='section-title'>相关论文</Text>
            <View className='related-papers'>
              {currentPaper.relatedPapers.map(paper => (
                <View
                  key={paper.id}
                  className='related-paper'
                  onClick={() => handleRelatedPaperClick(paper.id)}
                >
                  <Text className='title'>{paper.title}</Text>
                  <Text className='authors'>{paper.authors.join(', ')}</Text>
                </View>
              ))}
            </View>
          </View>
        )}
      </View>

      <View className='actions'>
        <CustomButton
          type='primary'
          block
          onClick={handleViewPaper}
        >
          查看原文
        </CustomButton>
      </View>
    </View>
  )
}

export default PaperDetail 