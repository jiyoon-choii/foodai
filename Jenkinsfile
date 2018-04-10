pipeline {
    agent Docker
    stages {
        stage('Run') {
            agent any
            steps {
                sh 'python foodai/foodai.py'
            }
        }
    }
}