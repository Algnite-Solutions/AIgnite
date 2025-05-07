import { View, Text, Radio, Input } from '@tarojs/components'
import { useState } from 'react'
import Taro from '@tarojs/taro'
import Form, { FormItem } from '../../components/ui/Form'
import CustomButton from '../../components/ui/Button'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import { setFrequency, registerStart, registerSuccess, registerFailure } from '../../store/slices/userSlice'
import './index.scss'

const Register = () => {
  const dispatch = useAppDispatch()
  const { frequency, loading, error } = useAppSelector(state => state.user)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [validationError, setValidationError] = useState('')

  const handleFrequencyChange = (e: any) => {
    dispatch(setFrequency(e.detail.value))
  }

  const handleUsernameChange = (e: any) => {
    setUsername(e.detail.value)
  }

  const handlePasswordChange = (e: any) => {
    setPassword(e.detail.value)
  }

  const handleConfirmPasswordChange = (e: any) => {
    setConfirmPassword(e.detail.value)
  }

  const validateForm = () => {
    if (!username.trim()) {
      setValidationError('用户名不能为空')
      return false
    }
    
    if (password.length < 6) {
      setValidationError('密码长度不能少于6位')
      return false
    }

    if (password !== confirmPassword) {
      setValidationError('两次输入的密码不一致')
      return false
    }

    setValidationError('')
    return true
  }

  const handleRegister = async () => {
    if (!validateForm()) {
      return
    }

    try {
      dispatch(registerStart())
      // TODO: 实现注册逻辑
      await new Promise(resolve => setTimeout(resolve, 1000)) // 模拟API调用
      dispatch(registerSuccess())
      Taro.showToast({
        title: '注册成功',
        icon: 'success',
        duration: 2000
      })
      Taro.navigateTo({
        url: '/pages/interests/index'
      })
    } catch (err) {
      dispatch(registerFailure(err instanceof Error ? err.message : '注册失败'))
      Taro.showToast({
        title: '注册失败',
        icon: 'error',
        duration: 2000
      })
    }
  }

  return (
    <View className='register-container'>
      <View className='header'>
        <Text className='title'>注册新账号</Text>
        <Text className='subtitle'>智能论文推荐助手</Text>
      </View>

      <Form>
        <FormItem label='用户名'>
          <Input
            type='text'
            placeholder='请输入用户名'
            value={username}
            onInput={handleUsernameChange}
            className='input'
          />
        </FormItem>

        <FormItem label='密码'>
          <Input
            password
            placeholder='请输入密码'
            value={password}
            onInput={handlePasswordChange}
            className='input'
          />
        </FormItem>

        <FormItem label='确认密码'>
          <Input
            password
            placeholder='请再次输入密码'
            value={confirmPassword}
            onInput={handleConfirmPasswordChange}
            className='input'
          />
        </FormItem>

        <FormItem label='选择订阅频率'>
          <View className='radio-group'>
            <View className='radio-item'>
              <Radio value='daily' checked={frequency === 'daily'} onClick={() => dispatch(setFrequency('daily'))}>每日推送</Radio>
            </View>
            <View className='radio-item'>
              <Radio value='weekly' checked={frequency === 'weekly'} onClick={() => dispatch(setFrequency('weekly'))}>每周推送</Radio>
            </View>
          </View>
        </FormItem>

        {validationError && <Text className='error-message'>{validationError}</Text>}
        {error && <Text className='error-message'>{error}</Text>}

        <CustomButton
          type='primary'
          block
          onClick={handleRegister}
          disabled={loading}
        >
          {loading ? '注册中...' : '注册'}
        </CustomButton>
        
        <View className='login-hint'>
          <Text>已有账号？</Text>
          <Text 
            className='login-link'
            onClick={() => Taro.navigateBack()}
          >
            返回登录
          </Text>
        </View>
      </Form>
    </View>
  )
}

export default Register 