package kotlin

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.JavaExec
import org.gradle.api.tasks.testing.TestDescriptor
import org.gradle.api.tasks.testing.TestResult

/**
 * A plugin for projects that contain Kotlin code. Safe to use with either of the ANX Java plugins.
 */
class KotlinPlugin implements Plugin<Project> {
    @Override
    void apply(Project project) {
        project.apply plugin: "kotlin"

        project.repositories {
            jcenter()
        }

        project.test {
            useTestNG()
            enableAssertions = true
            testLogging {
                showExceptions = true
                showStandardStreams = true
                exceptionFormat = 'full'
            }
        }

        def testSuccess = 0
        def failedTests = []
        project.test.afterTest {TestDescriptor tDesc, TestResult tResult ->
            assert tResult.testCount == 1
            if (tResult.successfulTestCount > 0) {
                ++testSuccess
            } else {
                failedTests << "${tDesc.className}#${tDesc.name}"
            }
        }

        project.getGradle().buildFinished {
            def totalTests = testSuccess + failedTests.size()
            if (totalTests > 0) {
                project.logger.error("Ran $totalTests tests for project ${project.path}. ${failedTests.size()} failed:")
                for (ft in failedTests) {
                    project.logger.error(": $ft")
                }
            }
        }

        KotlinLint.setupKtLint(project)
    }
}
