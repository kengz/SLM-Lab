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
        pretend: false
      }
    },
    watch: {
      lab_data: {
        files: `${dataSrc}/**`,
        tasks: ['sync'],
        options: {
          debounceDelay: 20 * 60 * 1000,
          interval: 60000,
        },
      }
    },
  })

  grunt.registerTask('default', ['watch'])
}
