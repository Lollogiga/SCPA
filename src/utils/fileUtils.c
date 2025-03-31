#include "../include/fileUtils.h"

#include "../include/constants.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>

int checkFolder(char *checkFolder, char **destFolder) {
    struct stat info;

    if (checkFolder && stat(checkFolder, &info) == 0 && S_ISDIR(info.st_mode)) {
        char *folder = malloc(strlen(checkFolder) + 1);
        if (!folder) {
            perror("checkFolder: error allocating space to forder path");
            return -1;
        }

        strcpy(folder, checkFolder);

        *destFolder = folder;

        return 0;
    }

#ifdef TEST
    if (stat(MATRIX_TEST_FOLDER_DEFAULT, &info) == 0 && (info.st_mode & S_IFDIR)) {
        char *folder = malloc(strlen(MATRIX_TEST_FOLDER_DEFAULT) + 1);
        if (!folder) {
            perror("checkFolder: error allocating space to forder path");
            return -1;
        }

        strcpy(folder, MATRIX_TEST_FOLDER_DEFAULT);

        *destFolder = folder;

        return 0;
    }
#endif

    if (stat(MATRIX_FOLDER_DEFAULT, &info) == 0 && (info.st_mode & S_IFDIR)) {
        char *folder = malloc(strlen(MATRIX_FOLDER_DEFAULT) + 1);
        if (!folder) {
            perror("checkFolder: error allocating space to forder path");
            return -1;
        }

        strcpy(folder, MATRIX_FOLDER_DEFAULT);

        *destFolder = folder;

        return 0;
    }

    return 1;
}