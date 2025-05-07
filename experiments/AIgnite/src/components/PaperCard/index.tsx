import { Text, View, Image } from '@tarojs/components'
import { useState } from 'react'
import Card from '../ui/Card'
import './index.scss'

export interface Paper {
  id: string
  title: string
  authors: string[]
  abstract: string
  hasInterpretation: boolean
  likes?: number
  dislikes?: number
}

interface PaperCardProps {
  paper: Paper
  onClick?: () => void
  onLike?: (id: string) => void
  onDislike?: (id: string) => void
}

const PaperCard: React.FC<PaperCardProps> = ({ paper, onClick, onLike, onDislike }) => {
  const [liked, setLiked] = useState(false)
  const [disliked, setDisliked] = useState(false)
  
  const handleLike = (e) => {
    e.stopPropagation()
    if (disliked) {
      setDisliked(false)
    }
    setLiked(!liked)
    if (onLike) {
      onLike(paper.id)
    }
  }
  
  const handleDislike = (e) => {
    e.stopPropagation()
    if (liked) {
      setLiked(false)
    }
    setDisliked(!disliked)
    if (onDislike) {
      onDislike(paper.id)
    }
  }

  return (
    <Card className='paper-card' onClick={onClick}>
      <Text className='title'>{paper.title}</Text>
      <Text className='authors'>{paper.authors.join(', ')}</Text>
      <Text className='abstract' numberOfLines={2}>
        {paper.abstract}
      </Text>
      
      <View className='paper-actions'>
      {paper.hasInterpretation && (
        <View className='interpretation-tag'>
          <Text>已生成解读</Text>
        </View>
      )}
        
        <View className='reaction-buttons'>
          <View className={`reaction-btn like-btn ${liked ? 'active' : ''}`} onClick={handleLike}>
            <Image 
              className='icon'
              src={require('../../assets/icons/like.png')}
            />
            <Text className='count'>{paper.likes || 0}</Text>
          </View>
          
          <View className={`reaction-btn dislike-btn ${disliked ? 'active' : ''}`} onClick={handleDislike}>
            <Image 
              className='icon'
              src={require('../../assets/icons/dislike.png')}
            />
            <Text className='count'>{paper.dislikes || 0}</Text>
          </View>
        </View>
      </View>
    </Card>
  )
}

export default PaperCard 