package publish

import org.gradle.api.InvalidUserDataException
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.publish.maven.MavenPublication

/**
 * Sets up default options for publishing our artifacts to bintray (from there they get synced to jcenter and Maven
 * central). For this to work the user must have the environment variables BINTRAY_USER and BINTRAY_KEY (the latter
 * being the Bintray API key) set.
 *
 * The version information comes from the VERSION file in the root of each subproject applying the plugin.
 */
class JavaPublishPlugin implements Plugin<Project> {
    private static String VERSION_FILE_NAME = 'VERSION'
    private static String MAVEN_GROUP = 'com.analyticspot.ml'

    @Override
    void apply(Project project) {
        project.with {
            apply plugin: 'maven-publish'
            apply plugin: 'com.jfrog.bintray'

            def version = rootProject.file(VERSION_FILE_NAME).text.trim()

            group = MAVEN_GROUP
            version = "$version"

            publishing {
                publications {
                    JarPublication(MavenPublication) {
                        from components.java
                        groupId MAVEN_GROUP
                        artifactId project.name
                        setVersion(version)
                    }
                }
            }

            bintray {
                user = System.getenv('BINTRAY_USER')
                key = System.getenv('BINTRAY_KEY')
                publications = ['JarPublication']
                // Makes it public so you don't have to go to bintray and manually publish it after upload.
                publish = true

                pkg {
                    repo = 'ANX'
                    name = 'ANX-ML-Framework'
                    licenses = ['LGPL-3.0']
                    vcsUrl = 'https://gitlab.com/oliver5/anxml'
                    websiteUrl = 'https://gitlab.com/oliver5/anxml'
                }
            }

            bintray.pkg.version {
                name = "$version"
                released = new Date()

                gpg {
                    sign = true
                }
            }

            // Make sure the right environment variables have been defined for pushing artifacts and fail if not.
            task('ensureBintrayEnv') {
                doLast {
                    def env1 = System.env.BINTRAY_KEY
                    def env2 = System.env.BINTRAY_USER
                    if (env1 == null || env1.isEmpty() || env2 == null || env2.isEmpty()) {
                        throw new InvalidUserDataException(
                                "You must provide environment variables BINTRAY_KEY and BINTRAY_USER in order to run" +
                                "the bintrayUpload task: https://github.com/bintray/gradle-bintray-plugin")
                    }
                }
            }

            tasks.bintrayUpload.dependsOn tasks.ensureBintrayEnv
        }
    }
}
