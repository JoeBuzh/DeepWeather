var path = require('path');

module.exports = {
  entry: './app.js',
  output: {
    filename: './static/bundle.js',
  },
  module: {
	exclude: ["./node_modules/"],
    loaders: [
      //转化ES6语法
      {
        test: /\.js$/,
        loader: 'babel',
        exclude: /node_modules/
      },
      //解析.vue文件
      {
        test:/\.vue$/,
        loader:'vue'
      },
      //图片转化，小于8K自动转化为base64的编码
      {
        test: /\.(png|jpg|gif)$/,
        loader:'url-loader?limit=8192'
      }
    ]
  },
  vue:{
    loaders:{
      js:'babel'
    }
  },
  resolve: {
        // require时省略的扩展名，如：require('app') 不需要app.js
        extensions: ['', '.js', '.vue'],
        // 别名，可以直接使用别名来代表设定的路径以及其他
        alias: {
            filter: path.join(__dirname, './filters'),
            components: path.join(__dirname, './components')
        }
    }    

};
