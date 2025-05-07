import { View, Text, Input, Picker } from '@tarojs/components'
import { useState } from 'react'
import { useAppDispatch, useAppSelector } from '../../store/hooks'
import {
  setFrequency,
  saveInterestsStart,
  saveInterestsSuccess,
  saveInterestsFailure,
  setInterestDescription
} from '../../store/slices/userSlice'
import { addFavorite, removeFavorite } from '../../store/slices/favoritesSlice'
import { AtTabs, AtTabsPane, AtIcon, AtToast, AtTag } from 'taro-ui'
import CustomButton from '../../components/ui/Button'
import Card from '../../components/ui/Card'
import InterestPaperCard from '../../components/ui/InterestPaperCard'
import { Paper } from '../../components/ui/PaperCard'
import Taro from '@tarojs/taro'
import './index.scss'

// 模拟搜索论文API调用
const searchPapers = async (query: string): Promise<Paper[]> => {
  // 模拟API延时
  await new Promise(resolve => setTimeout(resolve, 500))
  
  // 模拟搜索结果
  return [
    {
      id: '1001',
      title: 'Attention Is All You Need',
      authors: ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar', 'Jakob Uszkoreit', 'Llion Jones', 'Aidan N. Gomez', 'Łukasz Kaiser', 'Illia Polosukhin'],
      abstract: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.',
      tags: ['Deep Learning', 'NLP', 'Transformer'],
      submittedDate: '12 June, 2017',
      comments: 'Published in: Advances in Neural Information Processing Systems 30'
    },
    {
      id: '1002',
      title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
      authors: ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee', 'Kristina Toutanova'],
      abstract: 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.',
      tags: ['Deep Learning', 'NLP', 'BERT'],
      submittedDate: '11 October, 2018',
      comments: 'Published in: Proceedings of NAACL-HLT 2019'
    },
    {
      id: '1003',
      title: 'GPT-4 Technical Report',
      authors: ['OpenAI'],
      abstract: 'We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document, using both publicly available data and data licensed from third-party providers.',
      tags: ['GPT-4', 'Large Language Models', 'Multimodal'],
      submittedDate: '27 March, 2023',
      comments: 'Technical Report'
    }
  ].filter(paper => 
    !query || 
    paper.title.toLowerCase().includes(query.toLowerCase()) || 
    paper.authors.some(author => author.toLowerCase().includes(query.toLowerCase())) ||
    paper.abstract.toLowerCase().includes(query.toLowerCase())
  )
}

// AI领域选项
const AI_DOMAINS = [
  { name: '自然语言处理 (NLP)', active: false },
  { name: '计算机视觉 (CV)', active: false },
  { name: '大型语言模型 (LLM)', active: false },
  { name: '机器学习 (ML)', active: false },
  { name: '深度学习 (DL)', active: false },
  { name: '强化学习 (RL)', active: false },
  { name: '生成式AI (Generative AI)', active: false },
  { name: '多模态学习 (Multimodal)', active: false },
  { name: '语音识别 (ASR)', active: false },
  { name: '推荐系统 (Recommender)', active: false },
  { name: '图神经网络 (GNN)', active: false },
  { name: '联邦学习 (Federated)', active: false },
  { name: '知识图谱 (Knowledge Graph)', active: false },
]

const Interests = () => {
  const dispatch = useAppDispatch()
  const {
    frequency,
    interests,
    loading,
    error
  } = useAppSelector(state => state.user)

  // 获取收藏的论文
  const favoritedPapers = useAppSelector(state => state.favorites.papers)
  
  const [currentTab, setCurrentTab] = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Paper[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [interestDescription, setInterestDescription] = useState(interests.description || '')
  const [showSuccessToast, setShowSuccessToast] = useState(false)
  const [aiDomains, setAiDomains] = useState(AI_DOMAINS)

  const frequencyOptions = ['每日 (Daily)', '每周 (Weekly)']
  
  // Tab页内容
  const tabList = [
    { title: '订阅设置' },
    { title: '兴趣配置' }
  ]
  
  // 处理订阅频率选择
  const handleFrequencyChange = (e) => {
    const value = e.detail.value
    dispatch(setFrequency(value === '0' ? 'daily' : 'weekly'))
  }
  
  // 处理论文选择 - 修改为收藏功能
  const handlePaperSelect = (paperId: string) => {
    // 查找要添加或移除的论文
    const paper = searchResults.find(p => p.id === paperId)
    if (!paper) return
    
    // 检查论文是否已在收藏中
    const isFavorited = favoritedPapers.some(p => p.id === paperId)
    
    if (isFavorited) {
      // 从收藏中移除
      dispatch(removeFavorite(paperId))
      Taro.showToast({
        title: '已从收藏移除',
        icon: 'none',
        duration: 1500
      })
    } else {
      // 添加到收藏
      dispatch(addFavorite(paper))
      Taro.showToast({
        title: '已添加到收藏',
        icon: 'success',
        duration: 1500
      })
    }
    
    // 添加震动反馈
    Taro.vibrateShort({ type: 'medium' })
  }
  
  // 处理兴趣描述变更
  const handleDescriptionChange = (e) => {
    const value = e.detail.value
    setInterestDescription(value)
  }
  
  // 处理搜索查询变更
  const handleSearchChange = (e) => {
    setSearchQuery(e.detail.value)
  }
  
  // 处理AI领域选择
  const handleDomainClick = (index: number) => {
    const updatedDomains = [...aiDomains]
    updatedDomains[index].active = !updatedDomains[index].active
    setAiDomains(updatedDomains)

    // 更新描述文本
    const selectedDomains = updatedDomains
      .filter(domain => domain.active)
      .map(domain => domain.name.split(' ')[0]) // 只获取中文部分
    
    if (selectedDomains.length > 0) {
      setInterestDescription(selectedDomains.join('、'))
  }
  }
  
  // 执行搜索
  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([])
      return
    }
    
    setIsSearching(true)
    try {
      const results = await searchPapers(searchQuery)
      setSearchResults(results)
    } catch (error) {
      Taro.showToast({
        title: '搜索失败',
        icon: 'none'
      })
    } finally {
      setIsSearching(false)
    }
  }
  
  // 保存兴趣设置
  const handleSave = async () => {
    try {
      // 开始保存过程
      dispatch(saveInterestsStart())

      // 在此处进行兴趣描述的存储
      dispatch(setInterestDescription(interestDescription))
      
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // 保存成功
      dispatch(saveInterestsSuccess())
      
      // 显示成功提示
      setShowSuccessToast(true)
      
      // 震动反馈
      Taro.vibrateShort({ type: 'medium' })
      
      // 2秒后返回上一页
      setTimeout(() => {
        setShowSuccessToast(false)
      Taro.navigateBack()
      }, 2000)
    } catch (err) {
      dispatch(saveInterestsFailure(err instanceof Error ? err.message : '保存失败'))
      Taro.showToast({
        title: '保存失败',
        icon: 'error'
      })
    }
  }

  // 检查论文是否已收藏
  const isPaperFavorited = (paperId: string): boolean => {
    return favoritedPapers.some(paper => paper.id === paperId)
  }

  return (
    <View className='interests-page'>
      <View className='page-header'>
        <Text className='page-title'>个人中心</Text>
      </View>

      {/* 订阅频率设置部分 */}
      <View className='section-header'>
        <Text className='section-header-title'>订阅设置</Text>
      </View>
      
      <Card className='frequency-card'>
        <View className='section-title'>
          <AtIcon value='calendar' size='18' color='#4A89DC' />
          <Text>选择您希望接收推荐论文的频率</Text>
        </View>
        
        <View className='frequency-picker'>
          <Text className='label'>推送频率</Text>
          <Picker 
            mode='selector' 
            range={frequencyOptions}
            onChange={handleFrequencyChange}
            value={frequency === 'daily' ? 0 : 1}
          >
            <View className='picker-value'>
              <Text>{frequency === 'daily' ? '每日 (Daily)' : '每周 (Weekly)'}</Text>
              <AtIcon value='chevron-right' size='16' color='#999' />
            </View>
          </Picker>
        </View>
        
        <View className='frequency-description'>
          {frequency === 'daily' && (
            <Text user-select className='description-text'>每天将为您精选推送最相关、最新的学术论文，帮助您紧跟研究前沿</Text>
          )}
          {frequency === 'weekly' && (
            <Text user-select className='description-text'>每周将为您整合推送本周最重要、最相关的学术论文，减少信息过载</Text>
          )}
        </View>
        
        <View className='frequency-tips'>
          <View className='tip-item'>
            <AtIcon value='alert-circle' size='14' color='#FFB443' />
            <Text>您可以随时在设置中修改推送频率</Text>
          </View>
        </View>
      </Card>

      {/* 兴趣配置部分 */}
      <View className='section-header'>
        <Text className='section-header-title'>兴趣配置</Text>
      </View>
      
      <Card className='interests-card'>
        <View className='section-title'>
          <AtIcon value='tag' size='18' color='#4A89DC' />
          <Text>描述您的研究兴趣</Text>
        </View>
        
        <View className='interest-description'>
          <View className='textarea-container'>
            <Input
              className='description-input'
              type='text'
              value={interestDescription}
              onInput={handleDescriptionChange}
              placeholder='描述您的研究兴趣，例如: 自然语言处理、计算机视觉...'
              maxlength={100}
            />
            <View className='char-count'>
              <Text user-select>{interestDescription.length}/100</Text>
            </View>
          </View>
        </View>

        <View className='ai-domains-container'>
          <View className='domains-title'>
            <AtIcon value='bookmark' size='16' color='#4A89DC' />
            <Text>选择AI领域</Text>
          </View>
          <View className='domains-tags'>
            {aiDomains.map((domain, index) => (
              <View 
                key={domain.name} 
                className={`domain-tag ${domain.active ? 'active' : ''}`}
                onClick={() => handleDomainClick(index)}
              >
                <Text>{domain.name}</Text>
              </View>
            ))}
          </View>
        </View>
        
        <View className='section-title search-title'>
          <AtIcon value='search' size='16' color='#DC4A4A' />
          <Text>添加arXiv论文</Text>
        </View>

        <View className='search-container'>
          <View className='search-input'>
            <AtIcon value='search' size='16' color='#999' className='search-icon' />
            <Input
              className='description-input'
              type='text'
              placeholder='输入论文的arXiv地址，例如: 2201.12345 或论文标题...'
              value={searchQuery}
              onInput={handleSearchChange}
              onConfirm={handleSearch}
              confirmType='search'
            />
          </View>
          
          {isSearching && (
            <View className='loading-text'>
              <AtIcon value='loading-2' size='18' color='#999' className='loading-icon' />
              <Text user-select>正在搜索中...</Text>
            </View>
          )}
          
          {!isSearching && searchResults.length > 0 && (
            <View className='search-results-container'>
              <View className='results-count'>
                <Text user-select>搜索结果 ({searchResults.length})</Text>
              </View>
              <View className='search-results'>
                {searchResults.map(paper => (
                  <InterestPaperCard
              key={paper.id}
              paper={paper}
                    selected={isPaperFavorited(paper.id)}
                    onSelect={handlePaperSelect}
            />
          ))}
              </View>
            </View>
          )}
          
          {!isSearching && searchQuery && searchResults.length === 0 && (
            <View className='no-results'>
              <AtIcon value='alert-circle' size='18' color='#999' />
              <Text user-select>未找到相关论文</Text>
            </View>
          )}
        </View>
      </Card>

      {error && (
        <View className='error-message'>
          <AtIcon value='alert-circle' size='16' color='#DC4A4A' />
          <Text>{error}</Text>
        </View>
      )}

      <View className='save-button'>
          <CustomButton
            type='primary'
            size='large'
            block
            onClick={handleSave}
          disabled={loading}
          >
          {loading ? '保存中...' : '保存设置'}
          </CustomButton>
        </View>
      
      {/* 成功提示 */}
      <AtToast
        isOpened={showSuccessToast}
        text='保存成功'
        icon='check-circle'
        hasMask
        status='success'
      />
    </View>
  )
}

export default Interests 