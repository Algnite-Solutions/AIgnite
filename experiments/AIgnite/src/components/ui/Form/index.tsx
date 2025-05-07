import { View, Text } from '@tarojs/components'
import './index.scss'

interface FormItemProps {
  label: string
  hint?: string
  children: React.ReactNode
}

export const FormItem: React.FC<FormItemProps> = ({ label, hint, children }) => {
  return (
    <View className='form-item'>
      <Text className='label'>{label}</Text>
      {hint && <Text className='hint'>{hint}</Text>}
      {children}
    </View>
  )
}

interface FormProps {
  children: React.ReactNode
  className?: string
}

const Form: React.FC<FormProps> = ({ children, className = '' }) => {
  return (
    <View className={`form ${className}`}>
      {children}
    </View>
  )
}

export default Form 