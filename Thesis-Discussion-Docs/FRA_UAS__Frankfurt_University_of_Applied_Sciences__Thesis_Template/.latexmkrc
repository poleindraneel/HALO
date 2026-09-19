add_cus_dep('acn', 'acr', 0, 'makeglossaries');
add_cus_dep('glo', 'gls', 0, 'makeglossaries');
$clean_ext .= ' acr acn alg glo gls glg ist xdy';
sub makeglossaries {
    my ($base_name, $path) = fileparse( $_[0] );
    return system "makeglossaries", "-d", $path, $base_name;
}