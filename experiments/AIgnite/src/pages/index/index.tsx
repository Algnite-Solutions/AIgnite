import { View, Text } from '@tarojs/components'
import { useAppSelector } from '../../store/hooks'
import CustomButton from '../../components/ui/Button'
import Taro from '@tarojs/taro'
import './index.scss'

const Index = () => {
  const { isRegistered } = useAppSelector(state => state.user)

  const handleLogin = () => {
    // 直接跳转到论文推荐页面
      Taro.switchTab({
        url: '/pages/recommendations/index'
      })
  }
  
  const handleRegister = () => {
    // 跳转到注册页面
      Taro.navigateTo({
        url: '/pages/register/index'
      })
  }

  return (
    <View className='index-page'>
      <View className='header'>
        <Text className='title'>AIgnite</Text>
        <Text className='subtitle'>智能论文推荐助手</Text>
      </View>

      <View className='content'>
        <Text className='description'>
          基于您的兴趣和阅读习惯，为您推荐最相关的学术论文
        </Text>
      </View>

      <View className='actions'>
        <CustomButton
          type='primary'
          size='large'
          onClick={handleLogin}
          className='login-button'
        >
          登录
        </CustomButton>
        
        <CustomButton
          type='default'
          size='large'
          onClick={handleRegister}
          className='register-button'
        >
          注册
        </CustomButton>
      </View>
    </View>
  )
}

export default Index
