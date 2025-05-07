import { View } from '@tarojs/components'
import './index.scss'

interface CardProps {
  children: React.ReactNode
  className?: string
  onClick?: () => void
}

const Card: React.FC<CardProps> = ({ children, className = '', onClick }) => {
  return (
    <View className={`card ${className}`} onClick={onClick}>
      {children}
    </View>
  )
}

export default Card 