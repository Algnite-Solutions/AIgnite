export default defineAppConfig({
  pages: [
    'pages/index/index',
    'pages/register/index',
    'pages/interests/index',
    'pages/recommendations/index',
    'pages/paper-detail/index',
    'pages/favorites/index'
  ],
  window: {
    backgroundTextStyle: 'light',
    navigationBarBackgroundColor: '#fff',
    navigationBarTitleText: 'AIgnite - 智能论文推荐',
    navigationBarTextStyle: 'black'
  },
  tabBar: {
    custom: false,
    color: '#888888',
    selectedColor: '#1296db',
    backgroundColor: '#ffffff',
    borderStyle: 'black',
    list: [
      {
        pagePath: 'pages/recommendations/index',
        text: '推荐',
        iconPath: 'assets/icons/paper.png',
        selectedIconPath: 'assets/icons/paper.png'
      },
      {
        pagePath: 'pages/favorites/index',
        text: '收藏',
        iconPath: 'assets/icons/heart.png',
        selectedIconPath: 'assets/icons/heart.png'
      },
      {
        pagePath: 'pages/interests/index',
        text: '个人',
        iconPath: 'assets/icons/person.png',
        selectedIconPath: 'assets/icons/person.png'
      }
    ]
  }
})
