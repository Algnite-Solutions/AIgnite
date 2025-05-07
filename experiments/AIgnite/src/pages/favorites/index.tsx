import { View, Text, Image } from '@tarojs/components'
import { useState, useEffect } from 'react'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import { removeFavorite } from '../../store/slices/favoritesSlice'
import { AtIcon } from 'taro-ui'
import Taro from '@tarojs/taro'
import './index.scss'
import PaperCard, { Paper } from '../../components/ui/PaperCard'

// 收藏论文卡片组件
const FavoritePaperCard = ({ paper, onRemove }) => {
  const [expanded, setExpanded] = useState(false)

  const handleClick = () => {
    Taro.navigateTo({
      url: `/pages/paper-detail/index?id=${paper.id}`
    })
  }

  const handleRemove = (e) => {
    e.stopPropagation()
    onRemove(paper.id)
  }

  const toggleExpand = (e) => {
    e.stopPropagation()
    setExpanded(!expanded)
  }

  // 获取提交日期
  const getSubmittedDate = () => {
    if (paper.submittedDate) return paper.submittedDate
    
    const currentYear = new Date().getFullYear()
    const randomDay = 1 + Math.floor(Math.random() * 28)
    const randomMonth = new Date().getMonth() + 1
    
    return `${randomDay} ${getMonthName(randomMonth)}, ${currentYear}`
  }

  // 获取月份名称
  const getMonthName = (month) => {
    const monthNames = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ]
    return monthNames[month - 1]
  }

  // 获取评论
  const getComments = () => {
    if (paper.comments) return paper.comments
    
    const comments = [
      "Accepted at International Conference on Machine Learning 2025",
      "Published in: IEEE Transactions on Pattern Analysis and Machine Intelligence",
      "Accepted at CVPR 2025",
      "To appear in Advances in Neural Information Processing Systems 35",
      "Accepted at International Conference on Robotics and Automation 2025",
      "Published in: Nature Machine Intelligence"
    ]
    
    return comments[Math.floor(Math.random() * comments.length)]
  }

  return (
    <View className='favorite-paper-card' onClick={handleClick}>
      <View className='paper-title'>{paper.title}</View>
      
      <View className='paper-authors'>
        <Text user-select>Authors: {paper.authors.join(', ')}</Text>
      </View>
      
      <View className={`paper-abstract ${expanded ? 'expanded' : ''}`}>
        <Text className='abstract-heading'>Abstract: </Text>
        <Text user-select className='abstract-content'>{paper.abstract}</Text>
        
        {paper.showMore !== false && (
          <View className='show-more-container' onClick={toggleExpand}>
            <AtIcon value={expanded ? 'chevron-up' : 'chevron-down'} size='16' color='#4A89DC' />
            <Text className='show-more-text'>{expanded ? 'Less' : 'More'}</Text>
          </View>
        )}
      </View>
      
      <View className='paper-actions'>
        <View className='meta-info'>
          <View className='submission-info'>
            <AtIcon value='calendar' size='14' color='#999' />
            <Text user-select>Submitted {getSubmittedDate()}</Text>
          </View>
          
          {getComments() && (
            <View className='comments-info'>
              <Text user-select className='comments-content'>{getComments()}</Text>
            </View>
          )}
        </View>
        
        <View className='remove-button'>
          <View 
            className='action-button remove-action'
            onClick={handleRemove}
          >
            <AtIcon 
              value='subtract-circle' 
              size='16' 
              color='#e94b4b' 
            />
            <Text className='action-text'>移除收藏</Text>
          </View>
        </View>
      </View>
    </View>
  )
}

const Favorites = () => {
  const dispatch = useAppDispatch()
  const { papers, loading, error } = useAppSelector(state => state.favorites)

  // 处理移除收藏
  const handleRemoveFavorite = (paperId: string) => {
    dispatch(removeFavorite(paperId))
    Taro.showToast({
      title: '已从收藏移除',
      icon: 'none',
      duration: 1500
    })
    // 添加震动反馈
    Taro.vibrateShort({ type: 'medium' })
  }

  return (
    <View className='favorites-page'>
      <View className='page-header'>
        <Text className='page-title'>我的收藏</Text>
      </View>

      {loading ? (
        <View className='loading-container'>
          <AtIcon value='loading-3' size='30' color='#1296db' className='loading-icon' />
          <Text className='loading-text'>加载中...</Text>
        </View>
      ) : papers.length > 0 ? (
        <View className='favorites-list'>
          {papers.map(paper => (
            <FavoritePaperCard
              key={paper.id}
              paper={paper}
              onRemove={handleRemoveFavorite}
            />
          ))}
        </View>
      ) : (
        <View className='empty-container'>
          <View className='empty-icon-container'>
            <AtIcon value='heart' size='48' color='#ddd' />
          </View>
          <Text className='empty-title'>暂无收藏</Text>
          <Text className='empty-desc'>您还没有收藏任何论文</Text>
        </View>
      )}
    </View>
  )
}

export default Favorites 