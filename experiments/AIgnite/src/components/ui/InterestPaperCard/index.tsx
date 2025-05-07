import { View, Text, Image } from '@tarojs/components'
import { AtIcon } from 'taro-ui'
import Taro from '@tarojs/taro'
import { useState } from 'react'
import { Paper } from '../PaperCard' // Reusing the Paper interface
import './index.scss'

interface InterestPaperCardProps {
  paper: Paper
  selected?: boolean
  onClick?: () => void
  onSelect?: (paperId: string) => void
}

const InterestPaperCard: React.FC<InterestPaperCardProps> = ({ paper, selected = false, onClick, onSelect }) => {
  const [expanded, setExpanded] = useState(false)

  const handleClick = () => {
    if (onClick) {
      onClick()
    } else {
      Taro.navigateTo({
        url: `/pages/paper-detail/index?id=${paper.id}`
      })
    }
  }

  const handleSelect = (e) => {
    e.stopPropagation()
    if (onSelect) {
      onSelect(paper.id)
    }
  }

  const toggleExpand = (e) => {
    e.stopPropagation()
    setExpanded(!expanded)
  }

  // 获取提交日期
  const getSubmittedDate = () => {
    if (paper.submittedDate) return paper.submittedDate;
    
    const currentYear = new Date().getFullYear();
    const randomDay = 1 + Math.floor(Math.random() * 28);
    const randomMonth = new Date().getMonth() + 1;
    
    return `${randomDay} ${getMonthName(randomMonth)}, ${currentYear}`;
  };

  // 获取月份名称
  const getMonthName = (month) => {
    const monthNames = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December'
    ];
    return monthNames[month - 1];
  };

  // 获取评论
  const getComments = () => {
    if (paper.comments) return paper.comments;
    
    const comments = [
      "Accepted at International Conference on Machine Learning 2025",
      "Published in: IEEE Transactions on Pattern Analysis and Machine Intelligence",
      "Accepted at CVPR 2025",
      "To appear in Advances in Neural Information Processing Systems 35",
      "Accepted at International Conference on Robotics and Automation 2025",
      "Published in: Nature Machine Intelligence"
    ];
    
    return comments[Math.floor(Math.random() * comments.length)];
  };

  return (
    <View className={`interest-paper-card ${selected ? 'selected' : ''}`} onClick={handleClick}>
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
        
        <View className='selection-button'>
          <View 
            className={`action-button ${selected ? 'remove-button' : 'add-button'}`}
            onClick={handleSelect}
          >
            <AtIcon 
              value={selected ? 'subtract-circle' : 'add-circle'} 
              size='16' 
              color={selected ? '#e94b4b' : '#4A89DC'} 
            />
            <Text className='action-text'>{selected ? '移除' : '添加'}</Text>
          </View>
        </View>
      </View>
    </View>
  )
}

export default InterestPaperCard 