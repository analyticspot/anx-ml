package kotlin

import org.gradle.api.Project
import org.gradle.api.tasks.JavaExec

public class KotlinLint {
    /**
     * Sets up ktlint to run on the given project. Note that this requires that the "lint" and "check" tasks have
     * already been defined so you usually want to call this near the end of your plugin's apply method.
     */
    public static void setupKtLint(Project project) {
        project.configurations {
            ktlint
        }

        project.dependencies {
            ktlint 'com.github.shyiko:ktlint:0.6.1'
        }

        def lintOutputFile = "${project.buildDir}/ktlint.out"

        project.task('lint', type:JavaExec) {
            main = 'com.github.shyiko.ktlint.Main'
            classpath = project.configurations.ktlint
            args 'src/**/*.kt'

            inputs.files project.fileTree(project.projectDir) {
                include 'src/**/*.kt'
            }
            outputs.file project.file(lintOutputFile)
        }.doFirst {
            project.buildDir.mkdirs()
            // Puts a "tee" (like the Unix command "tee") in the output stream so the lint results are written
            // to standard out and to a file. Writing to a file lets us have an outputs declaration for the task so
            // that up-to-date checks work properly.
            standardOutput = new org.apache.tools.ant.util.TeeOutputStream(
                    new FileOutputStream(lintOutputFile), System.out);
        }

        assert project.tasks.findByName('check') != null, 'Make sure a "check" task exists before calling this'
        project.check.dependsOn project.lint
    }
}
