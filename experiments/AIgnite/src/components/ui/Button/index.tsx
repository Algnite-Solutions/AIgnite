import { Button } from '@tarojs/components'
import './index.scss'

interface ButtonProps {
  type?: 'primary' | 'default'
  size?: 'large' | 'medium' | 'small'
  block?: boolean
  disabled?: boolean
  onClick?: () => void
  className?: string
  children: React.ReactNode
}

const CustomButton: React.FC<ButtonProps> = ({
  type = 'primary',
  size = 'medium',
  block = false,
  disabled = false,
  onClick,
  className = '',
  children
}) => {
  return (
    <Button
      className={`custom-button ${type} ${size} ${block ? 'block' : ''} ${disabled ? 'disabled' : ''} ${className}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </Button>
  )
}

export default CustomButton 