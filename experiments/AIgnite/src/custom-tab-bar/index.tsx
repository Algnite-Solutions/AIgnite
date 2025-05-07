import { useEffect, useState } from 'react'
import { View, Text, Image } from '@tarojs/components'
import Taro from '@tarojs/taro'
import './index.scss'

interface TabItem {
  pagePath: string
  text: string
  iconPath: string
  selectedIconPath: string
}

const tabList: TabItem[] = [
  {
    pagePath: '/pages/recommendations/index',
    text: '推荐',
    iconPath: require('../assets/icons/paper.png'),
    selectedIconPath: require('../assets/icons/paper.png')
  },
  {
    pagePath: '/pages/favorites/index',
    text: '收藏',
    iconPath: require('../assets/icons/heart.png'),
    selectedIconPath: require('../assets/icons/heart.png')
  },
  {
    pagePath: '/pages/interests/index',
    text: '个人',
    iconPath: require('../assets/icons/person.png'),
    selectedIconPath: require('../assets/icons/person.png')
  }
]

export default function CustomTabBar() {
  const [selected, setSelected] = useState(0)
  const [animateTab, setAnimateTab] = useState(-1)

  useEffect(() => {
    const handlePageShow = () => {
      const currentPage = Taro.getCurrentInstance().router?.path
      const index = tabList.findIndex(item => currentPage === item.pagePath || currentPage === item.pagePath.slice(1))
      if (index !== -1) {
        setSelected(index)
      }
    }

    Taro.eventCenter.on('PAGE_SHOW', handlePageShow)
    handlePageShow()

    return () => {
      Taro.eventCenter.off('PAGE_SHOW', handlePageShow)
    }
  }, [])

  const switchTab = (index: number, item: TabItem) => {
    if (selected === index) return
    setAnimateTab(index)
    setTimeout(() => {
      setAnimateTab(-1)
    }, 300)
    Taro.switchTab({ url: item.pagePath })
  }

  return (
    <View className='custom-tab-bar'>
      {tabList.map((item, index) => {
        const isSelected = selected === index
        return (
          <View
            key={index}
            className={`tab-item ${isSelected ? 'selected' : ''} ${animateTab === index ? 'animate' : ''}`}
            onClick={() => switchTab(index, item)}
          >
            <View className='tab-icon-wrapper'>
              <Image 
                className='tab-icon'
                src={isSelected ? item.selectedIconPath : item.iconPath}
              />
            </View>
            <Text className='tab-text'>{item.text}</Text>
          </View>
        )
      })}
    </View>
  )
} 