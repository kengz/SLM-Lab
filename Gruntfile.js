const resolve = require('resolve-dir')


module.exports = function(grunt) {
  process.env.NODE_ENV = process.env.PY_ENV || 'development'

  const config = require('config')
  const dataSrc = 'data'
  const dataDest = resolve(config.data_sync_dir)

  require('load-grunt-tasks')(grunt)

  grunt.initConfig({
    sync: {
      lab_data: {
        files: [{
          cwd: dataSrc,
          src: ['**'],
          dest: dataDest,
        }],
      }
    },
    watch: {
      lab_data: {
        files: `${dataSrc}/**`,
        tasks: ['sync'],
      }
    },
  })

  grunt.registerTask('default', ['watch'])
}
