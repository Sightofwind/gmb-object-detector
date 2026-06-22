#ifndef _CONFIG_FILE_API_
#define _CONFIG_FILE_API_

#define KEY_VALUE_MAX_LEN (1024) // max lenght of key and value on config file, include string-tail '\0'
#define FILENAME_MAX_LEN  (384) // max lenght of config file-name, include string-tail '\0'

// read & write config file
// string line format: "key=value\n"
// note-line begin with "#" or ";"
int config_file_init(const char* filename); // setting config file-name,  must by called before other functions
int config_file_get(const char* key, char* value, int value_size, const char* filename = NULL); // return: value length, NOT include string-tail '\0'
int config_file_set(const char* key, const char* value, const char* filename = NULL); // return: 0 failed, 1 succeeded
int config_file_getall(char* buffer, int buffer_size, const char* filename = NULL);
int config_file_setall(const char* buffer, const char* filename = NULL);

// helper function: read & write integer and string
int config_file_get_int(const char* key, int default_value, const char* filename = NULL);
const char* config_file_get_str(const char* key, char* value, int value_size, const char* default_value, const char* filename = NULL);

// helper function: remove un-used char in string header & tail
void string_remove_mark(char* arg);

#endif // _CONFIG_FILE_API_
