import { View, Text } from '@tarojs/components'
import React, { useEffect } from 'react'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import {
  fetchRecommendationsStart,
  fetchRecommendationsSuccess,
  fetchRecommendationsFailure,
  clearRecommendations
} from '../../store/slices/paperSlice'
import PaperCard from '../../components/ui/PaperCard'
import Taro from '@tarojs/taro'
import './index.scss'

const Recommendations = () => {
  const dispatch = useAppDispatch()
  const {
    recommendations,
    loading,
    error,
    hasMore,
    page
  } = useAppSelector((state: any) => state.paper)

  const fetchRecommendations = async (pageNum: number) => {
    try {
      dispatch(fetchRecommendationsStart())
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // 真实学术论文样例
      const mockPapers = [
        {
          id: '1',
          title: 'Sim2Real Transfer for Vision-Based Grasp Verification',
          authors: ['Pau Amargant', 'Peter Hönig', 'Markus Vincze'],
          abstract: 'This paper presents a novel approach for determining whether a robot has successfully grasped an object. Our method employs a two-stage architecture; first YOLO-based object detection model to detect and locate the robot\'s gripper and then a ResNet-based classifier determines the presence of an object. To address the limitations of real-world data capture, we introduce HSR-GraspSynth, a synthetic dataset designed to simulate realistic grasping scenarios.',
          tags: ['Robotics', 'Computer Vision', 'Grasp Verification'],
          submittedDate: '5 May, 2025',
          publishDate: 'May 2025',
          comments: 'Accepted at Austrian Robotics Workshop 2025'
        },
        {
          id: '2',
          title: 'CLOG-CD: Curriculum Learning based on Oscillating Granularity of Class Decomposed Medical Image Classification',
          authors: ['Asmaa Abbas', 'Mohamed Gaber', 'Mohammed M. Abdelsamea'],
          abstract: 'In this paper, we have also investigated the classification performance of our proposed method based on different acceleration factors and pace function curricula. We used two pre-trained networks, ResNet-50 and DenseNet-121, as the backbone for CLOG-CD. The results with ResNet-50 show that CLOG-CD has the ability to improve classification performance significantly.',
          tags: ['Medical Imaging', 'Curriculum Learning', 'Deep Learning'],
          submittedDate: '3 May, 2025',
          publishDate: 'May 2025',
          comments: 'Published in: IEEE Transactions on Emerging Topics in Computing'
        },
        {
          id: '3',
          title: 'Attention-Based Feature Fusion for Visual Odometry with Unsupervised Scale Recovery',
          authors: ['Liu Wei', 'Zhang Chen', 'Wang Mei'],
          abstract: 'We present a novel approach for visual odometry that integrates attention mechanisms to fuse features from multiple sources. Our method addresses the scale ambiguity problem in monocular visual odometry through an unsupervised learning framework. Experimental results on KITTI dataset demonstrate superior performance compared to existing methods.',
          tags: ['Visual Odometry', 'Attention Mechanism', 'Unsupervised Learning'],
          submittedDate: '28 April, 2025',
          publishDate: 'April 2025',
          comments: 'To appear in International Conference on Robotics and Automation 2025'
        },
        {
          id: '4',
          title: 'FedMix: Adaptive Knowledge Distillation for Personalized Federated Learning',
          authors: ['Sarah Johnson', 'David Chen', 'Michael Brown'],
          abstract: 'This paper introduces FedMix, a novel framework for personalized federated learning that employs adaptive knowledge distillation to balance model personalization and global knowledge sharing. Our approach dynamically adjusts the knowledge transfer between global and local models based on client data distribution characteristics.',
          tags: ['Federated Learning', 'Knowledge Distillation', 'Personalization'],
          submittedDate: '15 April, 2025',
          publishDate: 'April 2025',
          comments: 'Accepted at International Conference on Machine Learning 2025'
        }
      ]
      
      dispatch(fetchRecommendationsSuccess({
        papers: mockPapers,
        hasMore: pageNum < 3 // 模拟只有3页数据
      }))
    } catch (err) {
      dispatch(fetchRecommendationsFailure(err instanceof Error ? err.message : '获取推荐失败'))
    }
  }

  const handleLoadMore = () => {
    if (!loading && hasMore) {
      fetchRecommendations(page)
    }
  }

  const handlePaperClick = (paperId: string) => {
    Taro.navigateTo({
      url: `/pages/paper-detail/index?id=${paperId}`
    })
  }

  // 初始加载
  useEffect(() => {
    dispatch(clearRecommendations())
    fetchRecommendations(1)
  }, [])

  return (
    <View className='recommendations-page'>
      <View className='header'>
        <Text className='title'>推荐论文</Text>
      </View>

      <View className='papers-list'>
        {recommendations.map(paper => (
          <PaperCard
            key={paper.id}
            paper={paper}
            onClick={() => handlePaperClick(paper.id)}
          />
        ))}
      </View>

      {error && <Text className='error-message'>{error}</Text>}

      {loading && (
        <View className='loading'>
          <Text>加载中...</Text>
        </View>
      )}

      {!loading && hasMore && (
        <View className='load-more' onClick={handleLoadMore}>
          <Text>加载更多</Text>
        </View>
      )}

      {!loading && !hasMore && recommendations.length > 0 && (
        <View className='no-more'>
          <Text>没有更多了</Text>
        </View>
      )}

      {!loading && recommendations.length === 0 && (
        <View className='empty'>
          <Text>暂无推荐论文</Text>
        </View>
      )}
    </View>
  )
}

export default Recommendations 